"""The baryon-pair completion of the mesonic continuum (``KernelConfig.baryon_pair_completion``, 2026-11-04).

Off, the kernel is the committed one.  On, every mesonic family gains the octet and decuplet pairs whose (B, Q, S)
sum to its own; the committed channels keep their relative weights (the completion stays outside the resonance pairs'
share hold), and the decuplet pairs redistribute the octet pairs' share."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

from na49_mixing import baryon_pair_completion as BPC
from na49_mixing import cascade_decay as C
from na49_mixing.cascade_decay import _qn_of
from na49_mixing.flavor_weights import FlavorParams
from na49_mixing.kernel_config import KernelConfig, load_kernel

NESTED_ROOT = Path(__file__).resolve().parents[1]
R137 = KernelConfig(**json.loads((NESTED_ROOT / "config" / "runs" / "r137.json").read_text())["kernel"])
FLAVOR = FlavorParams(gamma_s_had=0.29, r_v=0.77, r_bbar=3.05, assoc_norm=0.93, bbar_continuation_only=True)


@pytest.fixture(scope="module")
def kernels():
    return {scope: load_kernel(dataclasses.replace(R137, baryon_pair_completion=scope), NESTED_ROOT)
            for scope in (None, "strange", "strange_neutral")}


def _family(kernel, family_id):
    return next(f for f in kernel[1] if str(f.family_id) == family_id)


def _channels(kernel, family_id, mass=4.0):
    cat, fams = kernel
    family = _family(kernel, family_id)
    state = C.CascadeState(momentum=(mass, 0.0, 0.0, 0.0), mass=mass, qn=tuple(family.qn), quantum={},
                           family_id=family_id, axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None,
                           exchange_ancestry=("completion_test",), depth=0)
    return C.enumerate_channels(state, cat, fams, flavor=FLAVOR)


def test_off_is_the_committed_kernel(kernels):
    assert R137.as_dict()["baryon_pair_completion"] is None
    assert kernels[None][1] == load_kernel(R137, NESTED_ROOT)[1]
    for family in kernels[None][1]:
        assert family.completion_pair_classes == () and family.completion_resonance_classes == ()
        assert family.baryon_pair_completion is None


def test_the_options_are_checked():
    with pytest.raises(ValueError, match="baryon_pair_completion"):
        dataclasses.replace(R137, baryon_pair_completion="hyperons")
    with pytest.raises(ValueError, match="baryon_pair_completion"):
        dataclasses.replace(R137, baryon_pair_completion="strange", xi_tower=None, xi_baryons=False)


def test_the_classes_carry_the_family_quantum_numbers(kernels):
    cat, fams = kernels["strange"]
    for family in fams:
        classes = family.completion_pair_classes + family.completion_resonance_classes
        if int(family.qn[0]) != 0:
            assert not classes
        for class_id in classes:
            q0, q1 = (_qn_of(cat, d) for d in C._parse_class(class_id))
            assert tuple(int(a) + int(b) for a, b in zip(q0, q1)) == tuple(int(v) for v in family.qn), class_id
            assert int(q0[2]) != 0 or int(q1[2]) != 0, class_id     # the strange scope
    neutral = _family(kernels["strange"], "mesonic_neutral")
    assert {"(3122,-3122)", "(3312,-3312)", "(3322,-3322)"} <= set(neutral.completion_pair_classes)
    assert "(3324,-3324)" in neutral.completion_resonance_classes
    strange = _family(kernels["strange"], "mesonic_strange")
    assert {"(3122,-2112)", "(3222,-2212)", "(3322,-3122)", "(3312,-3112)"} <= set(strange.completion_pair_classes)
    # the neutral-only scope leaves every other family as committed
    for family in kernels["strange_neutral"][1]:
        if str(family.family_id) != BPC.NEUTRAL_FAMILY:
            assert family == next(f for f in kernels[None][1] if f.family_id == family.family_id)


def test_the_committed_channels_keep_their_relative_weights(kernels):
    """Within the continuation channels (the set the topology cascades draw from while one is open) and within
    the direct pairs, every committed channel keeps its weight relative to the others."""
    off = {(c.daughters, c.kind): c.weight for c in _channels(kernels[None], "mesonic_neutral")}
    on = {(c.daughters, c.kind): c.weight for c in _channels(kernels["strange"], "mesonic_neutral")}
    assert set(off) < set(on)
    for continuing in (True, False):
        keys = [k for k in off if (k[1] == C.KIND_CONTINUUM) == continuing]
        ref = max(keys, key=off.get)
        for key in keys:
            assert on[key] / on[ref] == pytest.approx(off[key] / off[ref], rel=1e-12), key


def test_the_decuplet_pairs_hold_the_octet_share(kernels):
    cat, fams = kernels["strange"]
    octet_only = [dataclasses.replace(f, completion_resonance_classes=()) for f in fams]

    def completion_share(fams_, family_id):
        family = next(f for f in fams_ if str(f.family_id) == family_id)
        classes = family.completion_pair_classes + family.completion_resonance_classes
        state = C.CascadeState(momentum=(4.0, 0.0, 0.0, 0.0), mass=4.0, qn=tuple(family.qn), quantum={},
                               family_id=family_id, axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None,
                               exchange_ancestry=("completion_test",), depth=0)
        raw = C.enumerate_channels(state, cat, fams_)
        cont = [c for c in raw if c.kind == C.KIND_CONTINUUM]
        hyperon = [c for c in cont if any(d.startswith(("baryonic_strange", "anti_baryonic_strange", "baryonic_xi",
                                                        "anti_baryonic_xi")) or
                                          (d.startswith(C.CATALOG_FAMILY_PREFIX) and
                                           int(cat.get(int(d[len(C.CATALOG_FAMILY_PREFIX):])).B) != 0 and
                                           int(cat.get(int(d[len(C.CATALOG_FAMILY_PREFIX):])).S) != 0)
                                          for d in c.daughters)]
        assert classes
        return math.fsum(c.weight for c in hyperon) / math.fsum(c.weight for c in cont)

    for family_id in ("mesonic_neutral", "mesonic_strange"):
        assert completion_share(fams, family_id) == pytest.approx(completion_share(octet_only, family_id), rel=1e-9)


def test_the_hyperon_pairs_pay_gamma_s(kernels):
    """At the neutral family a Lambda Lambdabar continuation creates one s-sbar pair and an Xi Xibar continuation
    two, so the Xi Xibar channels sit about gamma_s below the Lambda Lambdabar ones at equal decay momentum."""
    ch = {c.daughters: c.weight for c in _channels(kernels["strange"], "mesonic_neutral", mass=6.0)}
    lam = ch[("baryonic_strange", "catalog:-3122")]
    xi = ch[("baryonic_xi_q0", "catalog:-3322")]
    assert 0.1 * FLAVOR.gamma_s_had < xi / lam < 1.5 * FLAVOR.gamma_s_had


def test_the_octet_members_leave_the_decuplet_out():
    cat, fams = load_kernel(dataclasses.replace(R137, baryon_pair_completion="strange_neutral",
                                                baryon_pair_completion_members="octet"), NESTED_ROOT)
    neutral = next(f for f in fams if str(f.family_id) == "mesonic_neutral")
    assert "(3122,-3122)" in neutral.completion_pair_classes and neutral.completion_resonance_classes == ()
    with pytest.raises(ValueError, match="baryon_pair_completion_members"):
        dataclasses.replace(R137, baryon_pair_completion_members="decuplet")


def test_the_universal_share_holds_the_neutral_pair_share():
    """At a 4 GeV strange mesonic state, the completion's pairs that create no s-sbar (Y Nbar) take the neutral
    family's committed pair share at that mass, among the continuation channels."""
    cat, fams = load_kernel(dataclasses.replace(R137, baryon_pair_completion="all", baryon_pair_completion_members="octet",
                                                baryon_pair_share="universal"), NESTED_ROOT)
    strange = next(f for f in fams if str(f.family_id) == "mesonic_strange")
    assert strange.pair_share_reference == BPC.NEUTRAL_FAMILY
    assert next(f for f in fams if str(f.family_id) == "mesonic_neutral").pair_share_reference is None
    state = C.CascadeState(momentum=(4.0, 0.0, 0.0, 0.0), mass=4.0, qn=tuple(strange.qn), quantum={},
                           family_id="mesonic_strange", axis=(0.0, 0.0, 1.0), alignment_label="isotropic", parent=None,
                           exchange_ancestry=("completion_test",), depth=0)
    target = C._reference_pair_shares(BPC.NEUTRAL_FAMILY, state, cat, fams)[True]
    cont = [c for c in C.enumerate_channels(state, cat, fams) if c.kind == C.KIND_CONTINUUM]
    anchor = math.fsum(c.weight for c in cont if C._pair_total_b(c.daughters, cat, fams) >= 2
                       and C._created_strangeness(c.daughters, strange.qn, cat, fams) == 0)
    assert 0.0 < target < 0.5
    assert anchor / math.fsum(c.weight for c in cont) == pytest.approx(target, rel=1e-9)
    with pytest.raises(ValueError, match="baryon_pair_share"):
        dataclasses.replace(R137, baryon_pair_share="universal")


def _state(family, mass, parent):
    return C.CascadeState(momentum=(mass, 0.0, 0.0, 0.0), mass=mass, qn=tuple(family.qn), quantum={},
                          family_id=str(family.family_id), axis=(0.0, 0.0, 1.0), alignment_label="isotropic",
                          parent=parent, exchange_ancestry=("completion_test",), depth=0 if parent is None else 1)


def _listing(state, kernel):
    cat, fams = kernel
    return [(c.daughters, c.kind, c.weight) for c in C.enumerate_channels(state, cat, fams, flavor=FLAVOR)]


def test_the_line_share_completes_only_the_clusters_a_baryonic_line_emits():
    """The "line" share (2026-11-04): at the charged and strange mesonic families the completion is offered only on a
    state whose parent is baryonic, at the universal share; a root or a state continued from a mesonic one keeps the
    vertex of the kernel without that family's completion (the strange_neutral scope), and the neutral family keeps
    its completion everywhere."""
    base = dict(baryon_pair_completion_members="octet")
    neutral_only = load_kernel(dataclasses.replace(R137, baryon_pair_completion="strange_neutral", **base), NESTED_ROOT)
    universal = load_kernel(dataclasses.replace(R137, baryon_pair_completion="all", baryon_pair_share="universal",
                                                **base), NESTED_ROOT)
    line = load_kernel(dataclasses.replace(R137, baryon_pair_completion="all", baryon_pair_share="line", **base),
                       NESTED_ROOT)
    fams = {str(f.family_id): f for f in line[1]}
    assert fams[BPC.NEUTRAL_FAMILY].completion_line_only is False
    assert fams["mesonic_charged"].completion_line_only is True and fams["mesonic_strange"].completion_line_only is True
    proton_line = _state(next(f for f in line[1] if int(f.qn[0]) == 1 and int(f.qn[1]) == 1 and int(f.qn[2]) == 0),
                         6.0, None)
    mesonic = _state(fams[BPC.NEUTRAL_FAMILY], 6.0, None)
    for family_id in ("mesonic_charged", "mesonic_strange"):
        family = fams[family_id]
        emitted = _state(family, 4.0, proton_line)
        assert C._emitted_by_baryonic_line(emitted)
        assert _listing(emitted, line) == _listing(_state(family, 4.0, proton_line), universal)
        assert _listing(emitted, line) != _listing(_state(family, 4.0, proton_line), neutral_only)
        for parent in (None, mesonic):
            state = _state(family, 4.0, parent)
            assert not C._emitted_by_baryonic_line(state)
            assert _listing(state, line) == _listing(state, neutral_only)
    for parent in (None, proton_line, mesonic):
        state = _state(fams[BPC.NEUTRAL_FAMILY], 4.0, parent)
        assert _listing(state, line) == _listing(state, universal) == _listing(state, neutral_only)


def test_the_charged_nucleon_scope_adds_the_charged_families_nucleon_pairs():
    """The "charged_nucleon" scope (2026-11-04): the neutral family's classes of "strange_neutral", the charged
    families' non-strange classes alone (p nbar at the positive family, n pbar at the negative one), and no class
    at a strange family."""
    base = dataclasses.replace(R137, baryon_pair_completion_members="octet")
    neutral_only = {str(f.family_id): f for f in load_kernel(dataclasses.replace(
        base, baryon_pair_completion="strange_neutral"), NESTED_ROOT)[1]}
    cat, fams = load_kernel(dataclasses.replace(base, baryon_pair_completion="charged_nucleon"), NESTED_ROOT)
    for family in fams:
        if int(family.qn[0]) != 0:
            continue
        fid = str(family.family_id)
        if fid == BPC.NEUTRAL_FAMILY:
            assert family.completion_pair_classes == neutral_only[fid].completion_pair_classes != ()
        elif int(family.qn[2]) != 0:
            assert family.completion_pair_classes == () and family.baryon_pair_completion is None
        elif family.completion_pair_classes:
            for class_id in family.completion_pair_classes:
                x, ybar = C._parse_class(class_id)
                assert _qn_of(cat, x)[2] == 0 and _qn_of(cat, ybar)[2] == 0, class_id
    charged = {str(f.family_id): f for f in fams}
    assert "(2212,-2112)" in charged["mesonic_charged"].completion_pair_classes
    assert all(int(c.strip("()").split(",")[0]) in (2212, 2112) for c in charged["mesonic_charged"].completion_pair_classes)
