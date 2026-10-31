"""The scope of the baryon-pair weight r_BBbar (2026-10-25, at the R115
optimum): under ``FlavorParams.bbar_continuation_only`` r_BBbar acts on
the pair steps that continue (a continuum daughter beside the antibaryon,
or its conjugate) and the direct pairs of a floor state keep their bare
two-body weight.  The probes at the R115 optimum put 0.59-0.62 of the
forward pbar at x_F 0.25-0.45 in the floor's direct pairs of early
neutral mesonic states (0.095-0.113 of a 2.0-2.3 GeV state's decays at
r_BBbar = 2.445, against 0.5-1.7% for e+e- -> p pbar), and the floor's
direct pairs x 0.5 gained -249 / -168 with the normalizations fitted.

- the exponents: a direct pair counts n_bbar = 0 under the option and 1
  without it; a continuing pair step counts 1 either way; a channel
  without a pair is untouched;
- the weights: under the option a floor vertex's direct pair keeps the
  share it has at r_BBbar = 1, and a continuing vertex is unchanged;
- the ledger: the reweighting of a floor vertex to another r_BBbar is
  exactly zero under the option and the keys follow the generation's
  exponents;
- the engine threads the spec's ``flavor_bbar_scope`` into FlavorParams
  and rejects other values.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from na49_mixing.cascade_decay import (
    ISOTROPIC_ALIGNMENT_LABEL,
    KIND_CONTINUUM,
    CascadeChannel,
    CascadeState,
    _family_by_id,
    enumerate_channels,
)
from na49_mixing.flavor_weights import (FlavorParams, apply_flavor_weights, flavor_exponents, ledger_log_weight,
                                        vertex_ledger_record)
from na49_mixing.kernel_config import KernelConfig, load_kernel

NESTED_ROOT = Path(__file__).resolve().parents[1]
ENGINE = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
RUNS = NESTED_ROOT / "config" / "runs"


def _engine(name):
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


@pytest.fixture(scope="module")
def kernel():
    r115 = json.loads((RUNS / "r115.json").read_text())
    return load_kernel(KernelConfig(**r115["kernel"]), NESTED_ROOT)


def _state(fams, family_id, mass):
    fam = _family_by_id(fams, family_id)
    return CascadeState(momentum=(mass, 0.0, 0.0, 0.0), mass=mass, qn=tuple(fam.qn), quantum={}, family_id=family_id,
                        axis=(0.0, 0.0, 1.0), alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None,
                        exchange_ancestry=(), depth=1)


def _channel(parent, kind, *daughters):
    return CascadeChannel(parent_id=parent, daughters=tuple(daughters), kind=kind, weight=1.0)


def _pair_share(channels, catalog, fams, qn):
    tot = math.fsum(c.weight for c in channels)
    return math.fsum(c.weight for c in channels if flavor_exponents(qn, c, catalog, fams).n_bbar) / tot


def test_the_exponents_count_a_direct_pair_only_without_the_option(kernel):
    catalog, fams = kernel[0], kernel[1]
    neutral = (0, 0, 0)
    direct = _channel("mesonic_neutral", "established_hadron", "catalog:2212", "catalog:-2212")
    step = _channel("mesonic_neutral", KIND_CONTINUUM, "baryonic_np", "catalog:-2212")
    pions = _channel("mesonic_neutral", "established_hadron", "catalog:211", "catalog:-211")
    assert flavor_exponents(neutral, direct, catalog, fams).n_bbar == 1
    assert flavor_exponents(neutral, direct, catalog, fams, bbar_continuation_only=True).n_bbar == 0
    assert flavor_exponents(neutral, step, catalog, fams).n_bbar == 1
    assert flavor_exponents(neutral, step, catalog, fams, bbar_continuation_only=True).n_bbar == 1
    for opt in (False, True):
        assert flavor_exponents(neutral, pions, catalog, fams, bbar_continuation_only=opt).n_bbar == 0
    # every other exponent of the direct pair is the same either way
    a = flavor_exponents(neutral, direct, catalog, fams)
    b = flavor_exponents(neutral, direct, catalog, fams, bbar_continuation_only=True)
    assert (a.n_ssbar, a.n_v, a.n_assoc, a.n_channel) == (b.n_ssbar, b.n_v, b.n_assoc, b.n_channel)


def test_the_option_is_a_bool():
    with pytest.raises(ValueError):
        FlavorParams(r_bbar=2.0, bbar_continuation_only=1)
    assert FlavorParams(r_bbar=2.0).bbar_continuation_only is False


def test_a_floor_vertex_keeps_its_bare_pair_share_and_a_continuing_vertex_is_unchanged(kernel):
    catalog, fams = kernel[0], kernel[1]
    floor = _state(fams, "mesonic_neutral", 2.2)
    bare = enumerate_channels(floor, catalog, fams, FlavorParams(gamma_s_had=0.24, r_v=1.02, r_bbar=1.0, assoc_norm=1.53))
    every = enumerate_channels(floor, catalog, fams, FlavorParams(gamma_s_had=0.24, r_v=1.02, r_bbar=2.445,
                                                                 assoc_norm=1.53))
    cont_only = enumerate_channels(floor, catalog, fams, FlavorParams(gamma_s_had=0.24, r_v=1.02, r_bbar=2.445,
                                                                     assoc_norm=1.53, bbar_continuation_only=True))
    assert not any(c.kind == KIND_CONTINUUM for c in every)   # 2.2 GeV is a floor state: direct channels only
    s_bare, s_every, s_cont = (_pair_share(ch, catalog, fams, floor.qn) for ch in (bare, every, cont_only))
    assert s_every > 2.0 * s_bare > 0.0
    assert s_cont == pytest.approx(s_bare, rel=1e-12)
    high = _state(fams, "mesonic_neutral", 3.5)
    a = enumerate_channels(high, catalog, fams, FlavorParams(r_bbar=2.445))
    b = enumerate_channels(high, catalog, fams, FlavorParams(r_bbar=2.445, bbar_continuation_only=True))
    cont_a = [c for c in a if c.kind == KIND_CONTINUUM]
    cont_b = [c for c in b if c.kind == KIND_CONTINUUM]
    za, zb = math.fsum(c.weight for c in cont_a), math.fsum(c.weight for c in cont_b)
    assert [c.daughters for c in cont_a] == [c.daughters for c in cont_b]
    for x, y in zip(cont_a, cont_b):   # the continuation set the cascade draws over: the same relative weights
        assert x.weight / za == pytest.approx(y.weight / zb, rel=1e-12)


def test_the_ledger_of_a_floor_vertex_does_not_reweight_under_the_option(kernel):
    catalog, fams = kernel[0], kernel[1]
    floor = _state(fams, "mesonic_neutral", 2.2)
    gen = FlavorParams(gamma_s_had=0.24, r_v=1.02, r_bbar=2.445, assoc_norm=1.53, bbar_continuation_only=True)
    pool = enumerate_channels(floor, catalog, fams, gen)
    pair = next(c for c in pool if set(c.daughters) == {"catalog:2212", "catalog:-2212"})
    rec = vertex_ledger_record(floor.qn, pool, pair, catalog, fams, gen)
    moved = FlavorParams(gamma_s_had=0.24, r_v=1.02, r_bbar=1.5, assoc_norm=1.53, bbar_continuation_only=True)
    assert ledger_log_weight([rec], gen, moved) == 0.0
    # without the option the same vertex reweights with r_BBbar
    gen_every = FlavorParams(gamma_s_had=0.24, r_v=1.02, r_bbar=2.445, assoc_norm=1.53)
    pool_every = enumerate_channels(floor, catalog, fams, gen_every)
    pair_every = next(c for c in pool_every if set(c.daughters) == {"catalog:2212", "catalog:-2212"})
    rec_every = vertex_ledger_record(floor.qn, pool_every, pair_every, catalog, fams, gen_every)
    assert ledger_log_weight([rec_every], gen_every, FlavorParams(gamma_s_had=0.24, r_v=1.02, r_bbar=1.5,
                                                                  assoc_norm=1.53)) < 0.0
    # a continuing vertex reweights under the option as without it
    high = _state(fams, "mesonic_neutral", 3.5)
    for opt in (False, True):
        g = FlavorParams(r_bbar=2.445, bbar_continuation_only=opt)
        cont = [c for c in enumerate_channels(high, catalog, fams, g) if c.kind == KIND_CONTINUUM]
        step = next(c for c in cont if flavor_exponents(high.qn, c, catalog, fams).n_bbar)
        rec_c = vertex_ledger_record(high.qn, cont, step, catalog, fams, g)
        w = ledger_log_weight([rec_c], g, FlavorParams(r_bbar=1.5, bbar_continuation_only=opt))
        if opt is False:
            w_every = w
    assert w == pytest.approx(w_every, rel=1e-12) and w < 0.0


def test_the_engine_threads_the_scope(tmp_path):
    E = _engine("run_engine_bbar_scope_test")
    spec = json.loads((RUNS / "r115.json").read_text())
    E.configure(RUNS / "r115.json")
    assert E.BBAR_CONTINUATION_ONLY is False
    assert E.flavor_params(E.GEN_FLAVOR).bbar_continuation_only is False
    spec["flavor_bbar_scope"] = "continuation"
    path = tmp_path / "r115_scope.json"
    path.write_text(json.dumps(spec))
    E.configure(path)
    assert E.BBAR_CONTINUATION_ONLY is True
    assert E.flavor_params(E.GEN_FLAVOR).bbar_continuation_only is True
    spec["flavor_bbar_scope"] = "floor"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        E.configure(path)


def test_the_r116_spec_is_r115_with_the_bbar_scope():
    r115, r116 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r115", "r116"))
    assert r116["flavor_bbar_scope"] == "continuation" and "flavor_bbar_scope" not in r115
    for block in ("kernel", "topology", "fit", "diffractive_split", "isovector", "fixed", "statistics",
                  "central_component", "single_diffraction_window", "systematic_variants"):
        assert r116[block] == r115[block], block
    assert [p["name"] for p in r116["parameters"]] == [p["name"] for p in r115["parameters"]]
    assert r116["generation_0"]["run"] == "r115_intact_t_break_2026-10-24"
    assert r116["generation_0"]["from_fit"] == [n for n in r115["generation_0"]["from_fit"] if n != "r_bbar"]
    assert "r_bbar" in r116["generation_0"]["judgment"]   # generated at the probes' continuation factor, named
    assert r116["seeds"]["raw_first"] > r115["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r115_intact_t_break_2026-10-24" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert r116["generation_0"]["branch_digest"] == report["digest"]
        gen0 = {p["name"]: p["gen0"] for p in r116["parameters"]}
        for name in r116["generation_0"]["from_fit"]:
            assert float(gen0[name]) == float(report["fitted"][name]), name
        assert 1.0 <= float(gen0["r_bbar"]) / float(report["fitted"]["r_bbar"]) <= 1.5


def test_the_r117_spec_is_the_r116_replica():
    r115, r116, r117 = (json.loads((RUNS / f"{r}.json").read_text()) for r in ("r115", "r116", "r117"))
    assert r117["flavor_bbar_scope"] == "continuation"
    for block in ("kernel", "topology", "fit", "diffractive_split", "isovector", "fixed", "statistics",
                  "central_component", "single_diffraction_window", "systematic_variants"):
        assert r117[block] == r116[block], block
    assert [p["name"] for p in r117["parameters"]] == [p["name"] for p in r116["parameters"]]
    assert r117["generation_0"]["run"] == "r116_bbar_scope_2026-10-25"
    assert r117["generation_0"]["from_fit"] == r115["generation_0"]["from_fit"]   # r_bbar from the fit again
    assert "judgment" not in r117["generation_0"]
    assert r117["seeds"]["raw_first"] > r116["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r116_bbar_scope_2026-10-25" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert r117["generation_0"]["branch_digest"] == report["digest"]
        gen0 = {p["name"]: p["gen0"] for p in r117["parameters"]}
        for name in r117["generation_0"]["from_fit"]:
            assert float(gen0[name]) == float(report["fitted"][name]), name
