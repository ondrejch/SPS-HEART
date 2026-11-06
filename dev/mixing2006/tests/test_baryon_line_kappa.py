"""The aligned continuing baryonic steps (2026-11-05; KernelConfig.baryon_line_kappa): the inverse CDF of
exp(kappa cos theta), the vertex topology the rule acts on, the option's checks, and the committed kernel untouched."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing import cascade_decay as C
from na49_mixing.kernel_config import KernelConfig, load_kernel

NESTED_ROOT = Path(__file__).resolve().parents[1]
R140 = KernelConfig(**json.loads((NESTED_ROOT / "config" / "runs" / "r140.json").read_text())["kernel"])


@pytest.mark.parametrize("kappa", [0.25, 1.0, 4.0, 40.0])
def test_the_aligned_cosine_follows_exp_kappa_cos(kappa):
    u = (np.arange(20000) + 0.5) / 20000
    c = np.array([C._aligned_cos(x, kappa) for x in u])
    assert c.min() >= -1.0 and c.max() <= 1.0
    assert C._aligned_cos(0.0, kappa) == pytest.approx(-1.0) and C._aligned_cos(1.0, kappa) == pytest.approx(1.0)
    expected_mean = 1.0 / math.tanh(kappa) - 1.0 / kappa
    assert c.mean() == pytest.approx(expected_mean, abs=2e-3)


def test_the_rule_acts_on_a_continuing_baryonic_step_only():
    cat, fams = load_kernel(R140, NESTED_ROOT)
    baryonic = next(f for f in fams if str(f.family_id) == "baryonic_np")
    mesonic = next(f for f in fams if str(f.family_id) == "mesonic_neutral")
    proton, pion = cat.get(2212), cat.get(111)
    assert C._continuing_baryon_slot([("family", None, baryonic), ("catalog", 111, pion)]) == 0
    assert C._continuing_baryon_slot([("catalog", 111, pion), ("family", None, baryonic)]) == 1
    assert C._continuing_baryon_slot([("catalog", 2212, proton), ("family", None, mesonic)]) is None
    assert C._continuing_baryon_slot([("catalog", 2212, proton), ("catalog", 111, pion)]) is None


def test_the_option_is_checked_and_reaches_the_families():
    for bad in (0.0, -1.0, 60.0, True, "1"):
        with pytest.raises(ValueError, match="baryon_line_kappa"):
            dataclasses.replace(R140, baryon_line_kappa=bad)
    _cat, fams = load_kernel(dataclasses.replace(R140, baryon_line_kappa=0.5), NESTED_ROOT)
    assert {f.baryon_line_kappa for f in fams} == {0.5}
    _cat, fams = load_kernel(R140, NESTED_ROOT)
    assert {f.baryon_line_kappa for f in fams} == {None}


def test_the_nonstrange_scope_leaves_the_strangeness_transfer_isotropic():
    cat, fams = load_kernel(R140, NESTED_ROOT)
    strange = next(f for f in fams if int(f.qn[0]) == 1 and int(f.qn[2]) == -1)
    pion, kaon = cat.get(211), cat.get(321)
    assert C._continuing_baryon_slot([("family", None, strange), ("catalog", 211, pion)], "nonstrange") == 0
    assert C._continuing_baryon_slot([("family", None, strange), ("catalog", 321, kaon)], "nonstrange") is None
    assert C._continuing_baryon_slot([("family", None, strange), ("catalog", 321, kaon)], "all") == 0
    with pytest.raises(ValueError, match="baryon_line_kappa_scope"):
        dataclasses.replace(R140, baryon_line_kappa=1.0, baryon_line_kappa_scope="strange")


def test_the_final_step_option_is_checked_and_reaches_the_families():
    for bad in (0.0, -1.0, 60.0, True):
        with pytest.raises(ValueError, match="baryon_line_kappa_final"):
            dataclasses.replace(R140, baryon_line_kappa_final=bad)
    _cat, fams = load_kernel(dataclasses.replace(R140, baryon_line_kappa_final=1.0), NESTED_ROOT)
    assert {f.baryon_line_kappa_final for f in fams} == {1.0}


def test_the_strange_transfer_slot_and_option():
    cat, fams = load_kernel(R140, NESTED_ROOT)
    strange = next(f for f in fams if int(f.qn[0]) == 1 and int(f.qn[2]) == -1)
    pion, kaon = cat.get(211), cat.get(321)
    assert C._strange_transfer_slot([("family", None, strange), ("catalog", 321, kaon)]) == 0
    assert C._strange_transfer_slot([("catalog", 321, kaon), ("family", None, strange)]) == 1
    assert C._strange_transfer_slot([("family", None, strange), ("catalog", 211, pion)]) is None
    with pytest.raises(ValueError, match="baryon_line_kappa_strange"):
        dataclasses.replace(R140, baryon_line_kappa_strange=0.0)
    _cat, fams = load_kernel(dataclasses.replace(R140, baryon_line_kappa_strange=1.0), NESTED_ROOT)
    assert {f.baryon_line_kappa_strange for f in fams} == {1.0}


def test_the_helicity_axis_takes_the_beam_angle_onto_the_motion():
    st = C.CascadeState(momentum=(5.0, 1.0, 0.0, 3.0), mass=4.8, qn=(1, 1, 0), quantum={}, family_id="baryonic_np",
                        axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None, exchange_ancestry=(), depth=1)
    along = C._onto_helicity((0.0, 0.0, 1.0), st)
    n = (1.0 / math.sqrt(10.0), 0.0, 3.0 / math.sqrt(10.0))
    assert along == pytest.approx(n, abs=1e-12)
    v = (0.6, 0.0, 0.8)
    w = C._onto_helicity(v, st)
    assert sum(x * x for x in w) == pytest.approx(1.0)
    assert sum(a * b for a, b in zip(w, n)) == pytest.approx(0.8)
    with pytest.raises(ValueError, match="baryon_line_kappa_axis"):
        dataclasses.replace(R140, baryon_line_kappa=1.0, baryon_line_kappa_axis="target")


def test_the_kaon_led_pair_slot_and_option():
    cat, _fams = load_kernel(R140, NESTED_ROOT)
    lam, sig, p, kaon, kstar, pion = cat.get(3122), cat.get(3222), cat.get(2212), cat.get(321), cat.get(313), cat.get(211)
    assert C._yk_meson_slot([("catalog", 3122, lam), ("catalog", 321, kaon)]) == 1
    assert C._yk_meson_slot([("catalog", 313, kstar), ("catalog", 3222, sig)]) == 0
    assert C._yk_meson_slot([("catalog", 2212, p), ("catalog", 211, pion)]) is None  # no strangeness
    assert C._yk_meson_slot([("catalog", 3122, lam), ("catalog", 211, pion)]) is None  # a non-strange meson
    for bad in (0.0, -1.0, 60.0, True):
        with pytest.raises(ValueError, match="baryon_line_kappa_kaon"):
            dataclasses.replace(R140, baryon_line_kappa_kaon=bad)
    _cat, fams = load_kernel(dataclasses.replace(R140, baryon_line_kappa_kaon=2.0), NESTED_ROOT)
    assert {f.baryon_line_kappa_kaon for f in fams} == {2.0}
