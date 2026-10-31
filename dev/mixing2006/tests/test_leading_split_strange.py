"""The strange established baryons in the leading-baryon split (2026-10-28, the forward Lambda and K+):
``KernelConfig.leading_split_strange`` False keeps the hyperons out of the split channels; absent or True, the split
is the committed one (the same channel subset, object for object)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

from na49_mixing import cascade_decay as C
from na49_mixing.kernel_config import KernelConfig, load_kernel

NESTED_ROOT = Path(__file__).resolve().parents[1]
R125 = KernelConfig(**json.loads((NESTED_ROOT / "config" / "runs" / "r125.json").read_text())["kernel"])


def _root(M, family_id="baryonic_np"):
    return C.CascadeState(momentum=(M, 0.0, 0.0, 0.0), mass=M, qn=(1, 1, 0), quantum={}, family_id=family_id,
                          axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None,
                          exchange_ancestry=("test",), depth=0)


@pytest.fixture(scope="module")
def kernels():
    return {"committed": load_kernel(R125, NESTED_ROOT),
            "true": load_kernel(dataclasses.replace(R125, leading_split_strange=True), NESTED_ROOT),
            "off": load_kernel(dataclasses.replace(R125, leading_split_strange=False), NESTED_ROOT)}


def test_the_option_validates_and_reaches_every_family(kernels):
    assert R125.as_dict()["leading_split_strange"] is None
    assert dataclasses.replace(R125, leading_split_strange=False).leading_split_strange is False
    for bad in ({"leading_split_strange": 0}, {"leading_split_strange": "false"}, {"leading_split_strange": 1.0}):
        with pytest.raises(ValueError, match="leading_split_strange"):
            dataclasses.replace(R125, **bad)
    with pytest.raises(ValueError, match="needs leading_split_fraction"):
        dataclasses.replace(R125, leading_split_fraction=None, leading_split_kappa=None, meson_split_fraction=None,
                            meson_split_slope=None, leading_split_strange=False)
    for label, value in (("committed", None), ("true", True), ("off", False)):
        _cat, fams = kernels[label]
        assert all(getattr(f, "leading_split_strange", None) is value for f in fams), label


def _split_subset(kernel, M, with_family=False):
    cat, fams = kernel
    state = _root(M)
    channels = C.enumerate_channels(state, cat, fams)
    out = C._leading_split(state, channels, cat, fams)
    assert out is not None
    return out if with_family else out[1]


def _established(cat, fams, channel):
    ids = [C._daughter_identity(cat, fams, d) for d in channel.daughters]
    return ids[0] if ids[0][0] == "catalog" else ids[1]


def test_the_hyperons_leave_the_split_channels_and_the_committed_split_is_unchanged(kernels):
    cat, fams = kernels["committed"]
    for M in (3.0, 4.5, 6.0, 8.0):
        committed = _split_subset(kernels["committed"], M)
        same = _split_subset(kernels["true"], M)
        off = _split_subset(kernels["off"], M)
        strange = [c for c in committed if int(_established(cat, fams, c)[2].S) != 0]
        # the (Y, K-like continuum) channels open above about 3.1 GeV (a hyperon beside the 2.0 GeV floor); at 4.5 GeV
        # and above they are 27 of the 47 split channels
        assert bool(strange) == (M > 3.5), f"the committed split at {M} GeV: {len(strange)} hyperon channels"
        key = lambda c: (tuple(c.daughters), float(c.weight))  # noqa: E731  (different kernels, equal channels)
        assert [key(c) for c in same] == [key(c) for c in committed]
        assert [key(c) for c in off] == [key(c) for c in committed if int(_established(cat, fams, c)[2].S) == 0]
        assert all(int(_established(cat, fams, c)[2].S) == 0 for c in off)
        # the hyperon channels take their share of the split's budget with them: the split fraction at the vertex is
        # F times the non-strange weight share, so the non-strange leading baryons are drawn at the committed rate
        fam_c, _ = _split_subset(kernels["committed"], M, with_family=True)
        fam_o, _ = _split_subset(kernels["off"], M, with_family=True)
        w_off, w_all = math.fsum(c.weight for c in off), math.fsum(c.weight for c in committed)
        if strange:
            assert w_off < w_all
            assert fam_o.leading_split_fraction == pytest.approx(fam_c.leading_split_fraction * w_off / w_all)
            assert fam_o.leading_split_fraction * 1.0 / w_off == pytest.approx(fam_c.leading_split_fraction / w_all)
        else:
            assert [key(c) for c in off] == [key(c) for c in committed]
            assert fam_o.leading_split_fraction == fam_c.leading_split_fraction
