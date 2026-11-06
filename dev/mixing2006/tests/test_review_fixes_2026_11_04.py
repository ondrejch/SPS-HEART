"""The fixes of the four code reviews of 2026-11-04 (qwen38fn, ds41f, muse13f, glm53f), each pinned where it acts."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from na49_mixing import cascade_decay as C
from na49_mixing import resonance_widths as RW
from na49_mixing.flavor_weights import FlavorParams, ledger_log_weight
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.quantum_numbers import quantum_numbers
from na49_mixing.resonance_decay import clebsch_gordan_squared
from na49_mixing.split_reweight import split_log_ratio
from na49_mixing.topology_remnants import _draw_kind

NESTED_ROOT = Path(__file__).resolve().parents[1]
R137 = KernelConfig(**json.loads((NESTED_ROOT / "config" / "runs" / "r137.json").read_text())["kernel"])


@pytest.fixture(scope="module")
def kernel():
    return load_kernel(R137, NESTED_ROOT)


def test_a_narrow_line_is_drawn_on_its_pole(kernel):
    """glm53f F6: the eta (1.31 keV) sat inside one 0.54 MeV cell of the uniform draw grid."""
    cat, _fams = kernel
    eta = cat.get(221)
    rng = np.random.default_rng(7)
    dm = RW._pole_mass_map(cat, eta)
    ms = np.array([RW.draw_mass(eta.mass_range, eta, dm, rng, catalog=cat) for _ in range(4000)])
    assert abs(np.median(ms) - eta.pole_mass) < 5e-6
    assert np.mean(np.abs(ms - eta.pole_mass) < 50 * eta.nominal_width) > 0.85   # the line, not a 400-width slab
    # a resolved line keeps the uniform grid (its draw table carries a float cell width)
    omega = cat.get(223)
    entry = RW._draw_entry(omega, RW._pole_mass_map(cat, omega), omega.mass_range[0], omega.mass_range[1], RW.BLATT_WEISSKOPF_BARRIER, cat)
    assert isinstance(entry[3], float)


def test_the_open_band_above_a_convolved_opening_draws(kernel):
    """glm53f F5: a window inside the daughter grid's first cell returned None and stopped the run."""
    cat, _fams = kernel
    rho0 = cat.get(113)
    ms, _rho, _w = RW._daughter_grid(cat, rho0, RW.BLATT_WEISSKOPF_BARRIER)
    rng = np.random.default_rng(1)
    for d in (1e-6, 1e-4, 1e-3):
        m = RW.draw_broad_daughter_mass(float(ms[0]) + 0.938272 + d, 0.938272, 1, rho0, cat, rng)
        assert m is not None and float(ms[0]) <= m <= float(ms[0]) + d


def test_the_reference_share_cache_checks_its_kernel(kernel):
    """glm53f F9: an id()-keyed entry of another family list is never served."""
    cat, fams = kernel
    state = C.CascadeState(momentum=(4.0, 0.0, 0.0, 0.0), mass=4.0, qn=(0, 0, 0), quantum={}, family_id="mesonic_neutral",
                           axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None, exchange_ancestry=("t",), depth=0)
    fams_list = list(fams)
    C._REFERENCE_SHARES[(id(fams_list), "mesonic_neutral", 4.0)] = (object(), {True: 0.999, False: 0.999})
    out = C._reference_pair_shares("mesonic_neutral", state, cat, fams_list)
    assert out[True] < 0.9


def test_the_kind_draw_never_takes_a_zero_weight_kind():
    """qwen38fn L20."""
    kind, lower, upper = _draw_kind({"a": 0.5, "b": 0.5, "z": 0.0}, 1.0 - 1e-17)
    assert kind == "b" and lower == 0.5
    assert _draw_kind({"a": 0.3, "b": 0.7}, 0.5)[0] == "b"


def test_the_split_shares_must_be_interior():
    """muse13f L6."""
    with pytest.raises(ValueError, match="split share"):
        split_log_ratio({}, (0.2, 0.045, 4.0, 4.0), (0.0, 0.045, 4.0, 4.0))


def test_the_clebsch_gordan_refuses_an_impossible_projection():
    """glm53f F15: <1, +1/2; 1, -1/2 | 1, 0>^2 returned 1/8."""
    assert clebsch_gordan_squared(1.0, 0, 1.0, 1, 1.0, -1) == 0.0
    assert clebsch_gordan_squared(1.5, 1, 0.5, 1, 1.0, 0) == pytest.approx(2.0 / 3.0)


def test_the_axial_isoscalars_need_the_light_scalars():
    """glm53f F10: a bare KeyError at the kernel load before."""
    with pytest.raises(ValueError, match="axial_isoscalars needs light_scalar_mesons"):
        dataclasses.replace(R137, pdg_nucleon_resonances=False, light_scalar_mesons=False, charged_scalar_pairs=False)


def test_the_neutral_three_body_hyperon_modes_refuse_a_charged_parent():
    """muse13f M1, glm53f F14."""
    from na49_mixing.pdg_hyperons import _split
    with pytest.raises(ValueError, match="neutral parent"):
        _split("Sig pi pi", 1.0, 2)
    assert len(_split("Sig pi pi", 1.0, 0)) == 3


def test_the_multistrange_quantum_numbers():
    """muse13f H1, glm53f F14."""
    assert tuple(quantum_numbers(3312).as_dict().values()) == (1, -1, -2)
    assert tuple(quantum_numbers(-3322).as_dict().values()) == (-1, 0, 2)
    assert tuple(quantum_numbers(3334).as_dict().values()) == (1, -1, -3)


def test_the_ledger_refuses_a_reweight_across_the_unencoded_knobs():
    """ds41f S3, glm53f F4."""
    a = FlavorParams(gamma_s_had=0.3)
    with pytest.raises(ValueError, match="hidden s-sbar"):
        ledger_log_weight([], a, dataclasses.replace(a, hidden_ssbar_content={333: 0.5}))
    with pytest.raises(ValueError, match="bbar_continuation_only"):
        ledger_log_weight([], a, dataclasses.replace(a, bbar_continuation_only=True))
    assert ledger_log_weight([], a, dataclasses.replace(a, gamma_s_had=0.4)) == 0.0


def test_a_channel_closed_at_the_pole_opens_above_its_threshold():
    """grok47: the a0(980) K Kbar had zero width at every mass (closed at the pole); the closed-channel rule opens it
    above the threshold, normalized half a width above it; off, the committed zero."""
    for on, expect_open in ((False, False), (True, True)):
        cat, _fams = load_kernel(dataclasses.replace(R137, closed_channel_reference=on), NESTED_ROOT)
        a0 = cat.get(9000111)
        ch = next(c for c in a0.decay_channels if set(c.daughters) == {321, -321})
        dm = RW._pole_mass_map(cat, a0)
        width = RW.channel_width(1.05, ch, a0, dm, catalog=cat)
        assert (width > 0.0) is expect_open
        assert RW.channel_width(0.95, ch, a0, dm, catalog=cat) == 0.0     # below the threshold


def test_the_gottfried_jackson_axis_keeps_the_drawn_angle():
    """grok47: the split's angle about the incoming baryon in the root's rest frame."""
    e = math.sqrt(9.0 + 0.25 + 36.0)
    state = C.CascadeState(momentum=(e, 0.5, 0.0, 6.0), mass=3.0, qn=(1, 1, 0), quantum={}, family_id="baryonic_np",
                           axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None, exchange_ancestry=("t",), depth=0)
    n = C._onto_gottfried_jackson((0.0, 0.0, 1.0), state)
    assert n[0] < -0.2 and n[2] > 0.9                                   # tilted against the root's transverse motion
    for theta, phi in ((0.3, 0.0), (1.1, 2.0), (2.5, 4.0)):
        d = (math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta))
        r = C._onto_gottfried_jackson(d, state)
        assert math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(r, n))))) == pytest.approx(theta, abs=1e-12)
    rest = dataclasses.replace(state, momentum=(3.0, 0.0, 0.0, 0.0))
    assert C._onto_gottfried_jackson((0.6, 0.0, 0.8), rest) == (0.6, 0.0, 0.8)
    with pytest.raises(ValueError, match="leading_split_axis"):
        dataclasses.replace(R137, leading_split_axis="helicity")
