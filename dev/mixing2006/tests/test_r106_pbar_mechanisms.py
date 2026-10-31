"""The R106 mechanism probes of 2026-10-21 (nested commit 44715da, card
TASK-20261021-r106-pbar-and-forward-piplus-mechanisms): every mechanism
defaults off, keeps the R106 run bit for bit when off (the baseline-tree
subprocess digests), and pins itself when switched on: the running-width
opening of the resonance-pair channels (P1a), the direct baryon pairs kept
in the [2.302, 2.987] GeV continuation gap at the bare p pbar share (P1b),
the S-wave L = 0 law on a baryon-pair creation step (P3), the
forward-hemisphere baryon rule on non-split B* -> B + M*_cont steps (P4),
the isospin-forbidden (f1, pi0) and (h1, pi0) classes dropped from the
I = 0 families under the AXIAL_NEUTRAL_I1_ONLY probe flag (P2), and the
opt-in forward-origin warm-cache equivalence register (T1).

Test tiers: ``test_the_r106_events_are_bit_identical_to_the_baseline_tree``
runs two simulated-event subprocesses (about one minute of duality warm-up
each) and ``test_events_conserve_with_all_mechanisms_on`` warms the pair
line shapes (tens of seconds); both belong on the slow-tier list once
measured.  The T1 register runs only with ``NA49_R106_T1=1``."""

from __future__ import annotations

import dataclasses
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import na49_mixing.axial_isoscalars as axial_isoscalars
from na49_mixing.cascade_decay import (
    ISOTROPIC_ALIGNMENT_LABEL,
    KIND_CONTINUUM,
    CascadeState,
    _family_by_id,
    continuing_pool,
    decay_momentum,
    draw_decay_with_pool,
    enumerate_channels,
)
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

NESTED_ROOT = Path(__file__).resolve().parents[1]
RUNS = NESTED_ROOT / "config" / "runs"
BASELINE_SHA = "cd616dd"
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics",
          "decay_channel", "decay_kinematics")
TAG = 21021
MECHANISM_FLAGS = ("resonance_pair_line_shapes", "direct_baryon_pairs_in_gap",
                   "baryon_pair_s_wave", "baryon_forward_hemisphere")
EVENT_CLASS_COUNTS = {"sd_projectile": (2, 2, 0), "sd_target": (2, 2, 0), "dd": (2, 2, 0),
                      "nd": (2, 2, 0)}
#: the pair classes pinned to the I = 0 mesonic_neutral family by the R106
#: kernel's baryon_resonance_pairs block (checked, not assumed)
BARE_PAIR_PREFIXES = ("2212", "-2212", "2112", "-2112")
RESONANCE_PAIR_TAGS = ("12212", "12112", "2212", "2112", "2216", "2116", "2214",
                       "2114", "2224", "102212", "102112")


def _parameters(spec, remnant_setup="intact"):
    """The nominal parameters of the completions register (the intact proton
    remnant for the sd/dd classes, the high-mass continuum for nd)."""
    weights = (
        {"intact_proton": 1.0, "low_mass_baryonic_resonance": 0.0,
         "high_mass_baryonic_continuum": 0.0} if remnant_setup == "intact"
        else {"intact_proton": 0.0, "low_mass_baryonic_resonance": 0.0,
              "high_mass_baryonic_continuum": 1.0})
    return TopologyNominalParams(epsilon=0.52, b_sd=6.43, epsilon_nd=0.247, nd_central=False,
                                 remnant_weights=weights, **spec["topology"])


def _state(family_id, mass, qn=(0, 0, 0), pz=0.0, depth=1):
    energy = math.sqrt(mass * mass + pz * pz)
    return CascadeState(momentum=(energy, 0.0, 0.0, pz), mass=float(mass),
                        qn=(int(qn[0]), int(qn[1]), int(qn[2])), quantum={},
                        family_id=family_id, axis=(0.0, 0.0, 1.0),
                        alignment_label=ISOTROPIC_ALIGNMENT_LABEL, parent=None,
                        exchange_ancestry=("probe",), depth=depth)


def _baryon_number(cat, fams, token) -> int:
    if token.startswith("catalog:"):
        return int(cat.get(int(token[len("catalog:"):])).B)
    return int(_family_by_id(fams, token).qn[0])


def _is_baryon_pair_channel(cat, fams, daughters) -> bool:
    """Whether the daughters carry >= 2 baryon units in all: a direct pair
    channel (the counting of cascade_decay._hold_pair_shares)."""
    return sum(abs(_baryon_number(cat, fams, d)) for d in daughters) >= 2


def _pair_share(channels, cat, fams) -> float:
    total = math.fsum(c.weight for c in channels)
    assert total > 0.0
    return math.fsum(c.weight for c in channels
                     if _is_baryon_pair_channel(cat, fams, c.daughters)) / total


def _rest_pz(state, daughter) -> float:
    """The daughter's pz in the parent's rest frame (the cascade's isotropic
    two-body kinematics boosted by the parent's motion)."""
    energy, _px, _py, pz = (float(x) for x in daughter.momentum)
    beta = float(state.momentum[3]) / float(state.momentum[0])
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    return gamma * (pz - beta * energy)


@pytest.fixture(scope="module")
def r106():
    return json.loads((RUNS / "r106.json").read_text())


@pytest.fixture(scope="module")
def kernels(r106):
    """The R106 kernel in the flag states of this register (one load per
    distinct config; the duality and pair tables build lazily at the first
    draw)."""
    base = load_kernel(KernelConfig(**r106["kernel"]), NESTED_ROOT)
    single = {key: load_kernel(KernelConfig(**{**r106["kernel"], key: True}), NESTED_ROOT)
              for key in MECHANISM_FLAGS}
    all_on = load_kernel(KernelConfig(**{**r106["kernel"],
                                         **{key: True for key in MECHANISM_FLAGS}}), NESTED_ROOT)
    return SimpleNamespace(
        base=base,
        stripped=(base[0], [dataclasses.replace(f, resonance_pair_classes=())
                            for f in base[1]]),
        p1a=single["resonance_pair_line_shapes"],
        p1b=single["direct_baryon_pairs_in_gap"],
        p3=single["baryon_pair_s_wave"],
        p4=single["baryon_forward_hemisphere"],
        all_on=all_on)


def test_the_mechanisms_default_off_and_validate(r106):
    """Every probe flag is a plain bool defaulting to False; the R106 run
    sets none of them, and a non-bool is refused."""
    for name in MECHANISM_FLAGS:
        cfg = KernelConfig()
        assert getattr(cfg, name) is False
        with pytest.raises(ValueError, match=rf"{name} must be a bool"):
            KernelConfig(**{name: 1})
        assert r106["kernel"].get(name, False) is False


def test_as_dict_carries_the_mechanism_switches():
    cfg = KernelConfig(**dict.fromkeys(MECHANISM_FLAGS, True))
    carried = cfg.as_dict()
    for name in MECHANISM_FLAGS:
        assert carried.get(name) is True, name


def test_pair_line_opening_and_the_held_share(kernels):
    """P1a: at M = 2.2 the resonance-pair channels stay closed by default
    (the bare p pbar and n nbar pairs only) and open under
    ``resonance_pair_line_shapes`` from the threshold-mass sum; the vertex
    pair share stays at the bare share in all three states, and the drawn
    pair masses stay inside their windows."""
    for state_key in ("base", "p1a"):
        cat, fams = getattr(kernels, state_key)
        state = _state("mesonic_neutral", 2.2)
        channels = enumerate_channels(state, cat, fams)
        pairs = [c for c in channels
                 if all(d.startswith("catalog:") for d in c.daughters)
                 and _is_baryon_pair_channel(cat, fams, c.daughters)]
        bare = {frozenset(("2212", "-2212")), frozenset(("2112", "-2112"))}
        for channel in pairs:
            marks = frozenset(d[len("catalog:"):] for d in channel.daughters)
            if state_key == "base":
                assert marks in bare, channel
        if state_key == "p1a":
            assert len(pairs) > len(bare)
        if state_key == "p1a":
            opened = [frozenset(d[len("catalog:"):] for d in c.daughters)
                      for c in pairs if "12212" in {d[len("catalog:"):]
                                                    for d in c.daughters}]
            assert opened and all(len(marks - bare) for marks in opened)
            pool = pairs
            for seed in range(300):
                _channel, daughters, _pools = draw_decay_with_pool(
                    state, pool, cat, fams,
                    np.random.default_rng(seed), np.random.default_rng(seed + 4000))
                assert math.fsum(float(d.mass) for d in daughters) <= 2.2 + 1e-9
                for daughter in daughters:
                    record = cat.get(int(daughter.family_id[len("catalog:"):]))
                    assert record.mass_range[0] <= float(daughter.mass) <= record.mass_range[1]
    # the held pair share: the base and the P1a states each hold the share of
    # the vertex with the resonance-pair classes stripped (the bare share)
    shares = {}
    for key in ("base", "p1a", "stripped"):
        cat, fams = getattr(kernels, key)
        state = _state("mesonic_neutral", 2.2)
        shares[key] = _pair_share(enumerate_channels(state, cat, fams), cat, fams)
    for key in ("base", "p1a"):
        assert abs(shares[key] - shares["stripped"]) < 1e-12, key
    assert abs(shares["base"] - 0.037633206868396) < 1e-9


def test_direct_baryon_pairs_in_the_continuation_gap(kernels):
    """P1b: at M = 2.6 the continuing pool joins the open direct baryon pairs
    at their bare share of the direct pool (``continuing_pool``); at M = 2.99
    the baryon-pair continuations are open and the pool is the plain
    continuum pool again; the base kernel keeps the continuation-only pool
    at both masses."""
    cat, fams = kernels.p1b
    base_cat, base_fams = kernels.base
    for probe_mass, expect_pairs in ((2.6, True), (2.99, False)):
        state = _state("mesonic_neutral", probe_mass)
        channels = enumerate_channels(state, cat, fams)
        pool = continuing_pool(state, channels, cat, fams)
        direct = [c for c in channels if c.kind != KIND_CONTINUUM]
        pool_pairs = [c for c in pool
                      if c.kind != KIND_CONTINUUM
                      and _is_baryon_pair_channel(cat, fams, c.daughters)]
        if expect_pairs:
            assert pool_pairs and all(c.kind != KIND_CONTINUUM for c in pool_pairs)
            assert all(c.kind == KIND_CONTINUUM for c in pool if c.kind == KIND_CONTINUUM)
            pool_pair_share = math.fsum(c.weight for c in pool_pairs) \
                / math.fsum(c.weight for c in pool)
            assert abs(pool_pair_share - _pair_share(direct, cat, fams)) < 1e-12
        else:
            assert pool_pairs == []
            base_pool = continuing_pool(
                state, enumerate_channels(state, base_cat, base_fams),
                base_cat, base_fams)
            assert [(c.daughters, c.weight) for c in pool] == \
                [(c.daughters, c.weight) for c in base_pool]
    # the bare baryon-pair continuations sit inside the 2.99 pool: the direct
    # pairs bow out exactly when they are open (the card's [2.302, 2.987] gap)
    state = _state("mesonic_neutral", 2.99)
    ch399 = enumerate_channels(state, cat, fams)
    pair_continuations = [c for c in ch399
                          if c.kind == KIND_CONTINUUM
                          and _is_baryon_pair_channel(cat, fams, c.daughters)]
    assert pair_continuations
    # the control: the base kernel keeps the continuation-only pool at 2.6
    state = _state("mesonic_neutral", 2.6)
    channels = enumerate_channels(state, base_cat, base_fams)
    assert all(c.kind == KIND_CONTINUUM
               for c in continuing_pool(state, channels, base_cat, base_fams))


def test_baryon_pair_s_wave_law(kernels):
    """P3: on a B = 0 -> baryonic-family creation step the daughter-mass
    density draw uses the plain q* law (L = 0) under ``baryon_pair_s_wave``;
    the mean emitted anti-baryon momentum lowers against the family's
    L = 1 barrier, and every other step draws bit for bit."""
    cat0, fams0 = kernels.base
    cat3, fams3 = kernels.p3
    state = _state("mesonic_neutral", 3.5)
    channels0 = enumerate_channels(state, cat0, fams0)
    channels3 = enumerate_channels(state, cat3, fams3)
    # the same open channels at the same weights: the S-wave law acts at the
    # daughter-mass draw, not on the channel weights
    assert [(c.daughters, c.kind, c.weight) for c in channels0] == \
        [(c.daughters, c.kind, c.weight) for c in channels3]
    pools0 = [c for c in channels0
              if any(not d.startswith("catalog:")
                     and abs(int(_family_by_id(fams0, d).qn[0])) == 1 for d in c.daughters)]
    pools3 = [c for c in channels3
              if any(not d.startswith("catalog:")
                     and abs(int(_family_by_id(fams3, d).qn[0])) == 1 for d in c.daughters)]
    assert len(pools0) == len(pools3) > 0
    emitted = {False: [], True: []}
    for seed in range(2000):
        for on, cat, fams, pool in ((False, cat0, fams0, pools0), (True, cat3, fams3, pools3)):
            _channel, drawn, _pools = draw_decay_with_pool(
                state, pool, cat, fams,
                np.random.default_rng(seed), np.random.default_rng(seed + 1000))
            baryons = [d for d in drawn if int(d.qn[0]) == 1]
            antibaryons = [d for d in drawn if int(d.qn[0]) == -1]
            assert baryons and antibaryons
            emitted[on].append(decay_momentum(3.5, float(baryons[0].mass),
                                              float(antibaryons[0].mass)))
    mean_off, mean_on = (float(np.mean(emitted[flag])) for flag in (False, True))
    assert mean_off - mean_on > 0.02, (mean_off, mean_on)
    # every other step keeps its draw: the baryonic parent (B != 0), the
    # anti-baryonic parent (the override needs B = 0) and the B = 0 ->
    # B = 0 (mesonic) steps
    for family_id, mass, qn in (("baryonic_np", 5.0, (1, 1, 0)),
                                ("anti_baryonic_np", 5.0, (-1, -1, 0))):
        other = _state(family_id, mass, qn=qn)
        pool0 = enumerate_channels(other, cat0, fams0)
        pool3 = enumerate_channels(other, cat3, fams3)
        for seed in range(50):
            out0 = draw_decay_with_pool(other, pool0, cat0, fams0,
                                        np.random.default_rng(seed),
                                        np.random.default_rng(seed + 77))
            out3 = draw_decay_with_pool(other, pool3, cat3, fams3,
                                        np.random.default_rng(seed),
                                        np.random.default_rng(seed + 77))
            assert [(d.mass, d.qn) for d in out0[1]] == [(d.mass, d.qn) for d in out3[1]]
    meson_state = _state("mesonic_neutral", 5.0)
    meson_pool0 = [c for c in enumerate_channels(meson_state, cat0, fams0)
                   if all(d.startswith("catalog:")
                          and abs(int(cat0.get(int(d[len("catalog:"):])).B)) == 0
                          for d in c.daughters)]
    meson_pool3 = [c for c in enumerate_channels(meson_state, cat3, fams3)
                   if all(d.startswith("catalog:")
                          and abs(int(cat3.get(int(d[len("catalog:"):])).B)) == 0
                          for d in c.daughters)]
    for seed in range(50):
        out0 = draw_decay_with_pool(meson_state, meson_pool0, cat0, fams0,
                                    np.random.default_rng(seed), np.random.default_rng(seed + 77))
        out3 = draw_decay_with_pool(meson_state, meson_pool3, cat3, fams3,
                                    np.random.default_rng(seed), np.random.default_rng(seed + 77))
        assert [(d.mass, d.qn) for d in out0[1]] == [(d.mass, d.qn) for d in out3[1]]


def test_baryon_forward_hemisphere_rule(kernels):
    """P4: at a non-split decay of a B = +/- 1 state into the established
    baryon and a mesonic continuum daughter, the emitted baryon-line daughter
    keeps the parent's hemisphere under ``baryon_forward_hemisphere`` (the
    anti-baryon on an anti-baryonic line as the charge conjugate of the
    baryon on a baryonic line), the isotropic rule
    stays exact with the flag off, and the mesonic parents are untouched."""
    cat0, fams0 = kernels.base
    cat4, fams4 = kernels.p4
    for family_id, qn in (("baryonic_np", (1, 1, 0)), ("anti_baryonic_np", (-1, -1, 0))):
        for side in (+1, -1):
            parent = _state(family_id, 5.0, qn=qn, pz=0.4 * side)
            pool = [c for c in enumerate_channels(parent, cat4, fams4)
                    if c.kind == KIND_CONTINUUM
                    and any(d.startswith("catalog:")
                            and int(cat4.get(int(d[len("catalog:"):])).B) * int(qn[0]) == 1
                            for d in c.daughters)]
            assert pool
            expected = float(side)
            fractions = {}
            for label, cat, fams in (("on", cat4, fams4), ("off", cat0, fams0)):
                forward = 0
                hits = 0
                for seed in range(200):
                    _channel, drawn, _pools = draw_decay_with_pool(
                        parent, pool, cat, fams,
                        np.random.default_rng(seed), np.random.default_rng(seed + 1))
                    for daughter in drawn:
                        if int(daughter.qn[0]) * int(qn[0]) != 1:
                            continue
                        hits += 1
                        forward += math.copysign(1.0, _rest_pz(parent, daughter)) == expected
                assert hits == 200
                fractions[label] = forward / hits
            assert fractions["on"] == 1.0, (family_id, side, fractions)
            assert 0.3 < fractions["off"] < 0.7, (family_id, side, fractions)
    # the mesonic parents are untouched: identical draws and identical
    # uniform consumption across the flag states
    for state in (_state("mesonic_neutral", 5.0, pz=0.0), _state("mesonic_neutral", 3.0, pz=0.3)):
        pool0 = enumerate_channels(state, cat0, fams0)
        pool4 = enumerate_channels(state, cat4, fams4)
        assert [(c.daughters, c.kind, c.weight) for c in pool0] == \
            [(c.daughters, c.kind, c.weight) for c in pool4]
        for seed in range(50):
            rc0, rk0 = np.random.default_rng(seed), np.random.default_rng(seed + 5)
            rc4, rk4 = np.random.default_rng(seed), np.random.default_rng(seed + 5)
            out0 = draw_decay_with_pool(state, pool0, cat0, fams0, rc0, rk0)
            out4 = draw_decay_with_pool(state, pool4, cat4, fams4, rc4, rk4)
            assert out0[:2] == out4[:2]
            assert rc0.bit_generator.state == rc4.bit_generator.state
            assert rk0.bit_generator.state == rk4.bit_generator.state


def test_the_axial_isoscalar_i1_only_probe(monkeypatch, r106, kernels):
    """P2: the (f1(1285), pi0) and (h1(1170), pi0) classes stay out of the
    I = 0 families under ``AXIAL_NEUTRAL_I1_ONLY`` while the I = 1 charged
    families keep their (f1, pi+/-) and (h1, pi+/-) classes."""
    assert axial_isoscalars.AXIAL_NEUTRAL_I1_ONLY is False
    cat, fams = kernels.base
    by_id = {f.family_id: f for f in fams}
    before = {f.family_id: set(f.allowed_decay_classes) for f in fams}
    assert {"(20223,111)", "(10223,111)"} <= before["mesonic_neutral"]
    assert {"(20223,211)", "(10223,211)"} <= before["mesonic_charged"]
    monkeypatch.setattr(axial_isoscalars, "AXIAL_NEUTRAL_I1_ONLY", True)
    _, fams_probe = load_kernel(KernelConfig(**r106["kernel"]), NESTED_ROOT)
    by_id_probe = {f.family_id: f for f in fams_probe}
    after = {f.family_id: set(f.allowed_decay_classes) for f in fams_probe}
    assert after["mesonic_neutral"]
    pi0_twins = {"(20223,111)", "(10223,111)"}
    assert not (pi0_twins & after["mesonic_neutral"])
    assert after["mesonic_neutral"] <= before["mesonic_neutral"] - pi0_twins
    for name in ("mesonic_charged", "mesonic_anticharged"):
        assert after[name] == before[name], (name, sorted(after[name] ^ before[name])[:4])
    assert axial_isoscalars.AXIAL_NEUTRAL_I1_ONLY is True  # monkeypatch is still live


# --------------------------------------------------------------- runners ---


_RUNNER = r'''
"""The simulated-event runner of the bit-identity register: one R106 event
record per requested (class, seed), serialized into a canonical JSON file."""
import dataclasses
import json
import sys
from pathlib import Path

package_root = Path(sys.argv[1])
out = Path(sys.argv[2])
sys.path.insert(0, str(package_root))
import numpy as np
from na49_mixing.kernel_config import KernelConfig, load_kernel
from na49_mixing.topology_runner import TopologyNominalParams, run_topology

spec = json.loads((package_root / "config" / "runs" / "r106.json").read_text())
LABELS = ("topology_selection", "spectral_state", "exchange_kinematics",
          "decay_channel", "decay_kinematics")
cat, fams = load_kernel(KernelConfig(**spec["kernel"]), package_root)
params = TopologyNominalParams(epsilon=0.52, b_sd=6.43, epsilon_nd=0.247, nd_central=False,
                               remnant_weights={"intact_proton": 1.0,
                                                "low_mass_baryonic_resonance": 0.0,
                                                "high_mass_baryonic_continuum": 0.0},
                               **spec["topology"])
TAG = 21021
EVENTS = [("sd_projectile", 1), ("sd_projectile", 2), ("sd_target", 0),
          ("sd_target", 1), ("dd", 0), ("dd", 2)]
records = []
for label, seed in EVENTS:
    rngs = {f"rng_{lab}": np.random.default_rng([seed, TAG, k])
            for k, lab in enumerate(LABELS)}
    ev = run_topology(label, params, cat, fams, **rngs)
    records.append({"class": label, "seed": seed, "event": dataclasses.asdict(ev)})
out.write_text(json.dumps(records, sort_keys=True))
'''


def _has_baseline_tree() -> bool:
    if shutil.which("git") is None:
        return False
    return subprocess.run(["git", "-C", str(NESTED_ROOT), "cat-file", "-e",
                           f"{BASELINE_SHA}^{{commit}}"],
                          capture_output=True).returncode == 0


def test_the_r106_events_are_bit_identical_to_the_baseline_tree(tmp_path):
    """The R106 run at every mechanism off: the six simulated events of the
    runner are byte-identical between the working tree and the baseline
    commit (cd616dd) extracted into a scratch tree (the default-off pin of
    the card; both trees run in clean subprocesses)."""
    if not _has_baseline_tree():
        pytest.skip("the baseline commit shape is not available in this checkout")
    assert NESTED_ROOT.parent.name == "dev", "the register assumes the monorepo layout"
    work = tmp_path / "baseline"
    work.mkdir()
    archive = subprocess.run(["git", "-C", str(NESTED_ROOT), "archive", BASELINE_SHA,
                              "na49_mixing", "config"], capture_output=True)
    assert archive.returncode == 0, archive.stderr.decode(errors="replace")
    tarfile.open(fileobj=io.BytesIO(archive.stdout)).extractall(work)
    runner = tmp_path / "runner.py"
    runner.write_text(_RUNNER)
    base_out = tmp_path / "base.json"
    head_out = tmp_path / "head.json"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    base = subprocess.run([sys.executable, str(runner), str(work), str(base_out)],
                          capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert base.returncode == 0, base.stderr[-2000:]
    head = subprocess.run([sys.executable, str(runner), str(NESTED_ROOT), str(head_out)],
                          capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert head.returncode == 0, head.stderr[-2000:]
    assert base_out.read_bytes() == head_out.read_bytes()
    for record in json.loads(head_out.read_text()):
        sums = tuple(sum(int(h["qn"][i]) for h in record["event"]["final_state"])
                     for i in range(3))
        assert sums == EVENT_CLASS_COUNTS[record["class"]], record


def test_events_conserve_with_all_mechanisms_on(r106, kernels):
    """With the four mechanisms switched on the simulated sd events still
    conserve (B, Q, S) exactly (the flags rearrange channels, laws and
    hemispheres; they never alter the quantum ledger)."""
    cat, fams = kernels.all_on
    params = _parameters(r106)
    for label, seed in (("sd_projectile", 1), ("sd_target", 0)):
        rngs = {f"rng_{lab}": np.random.default_rng([seed, TAG, k])
                for k, lab in enumerate(LABELS)}
        ev = run_topology(label, params, cat, fams, **rngs)
        sums = tuple(sum(int(h.qn[i]) for h in ev.final_state) for i in range(3))
        assert sums == EVENT_CLASS_COUNTS[label], (label, seed, sums)


def test_the_forward_origin_warm_cache_events_match_the_fresh_ones():
    """T1 (opt-in, ``NA49_R106_T1=1``): the forward-origin tool's warm-cache
    (memoized kernels, worker warm-up) and fresh-cache modes produce the
    same event counts and tally sums for the R106 spec."""
    tooling = NESTED_ROOT.parent.parent / ".collab" / "tooling" / "forward_origin.py"
    if os.environ.get("NA49_R106_T1") != "1" or not tooling.is_file():
        pytest.skip("the T1 register is opt-in (NA49_R106_T1=1) and the tooling "
                    "is not committed")
    assert NESTED_ROOT.parent.name == "dev"
    out_dir = NESTED_ROOT.parent / "00runs" / "tmp" / "r106_t1_register"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PROBE_SPEC="r106.json", PROBE_PROCS="2",
               PROBE_UNIT="12", PROBE_REPORT=str(out_dir / "report.json"))
    result = subprocess.run([sys.executable, str(tooling)], capture_output=True,
                            text=True, env=env, cwd=str(NESTED_ROOT.parent.parent))
    if result.returncode != 0:
        pytest.fail(f"the T1 warm/fresh register failed: {result.stderr[-2000:]}")
    report = json.loads((out_dir / "report.json").read_text())
    warm, fresh = report["warm"], report["fresh"]
    for unit in sorted(set(warm) | set(fresh)):
        assert unit in warm and unit in fresh, unit
        assert warm[unit]["events"] == fresh[unit]["events"], unit
        for tally in sorted(warm[unit]["tallies"]):
            assert abs(warm[unit]["tallies"][tally]
                       - fresh[unit]["tallies"][tally]) < 1e-15


def test_the_r107_spec_is_r106_with_the_pair_s_wave():
    """R107: the R106 contract with ``baryon_pair_s_wave`` on, generated at
    the sealed R106 optimum, on a seed block of its own."""
    runs = NESTED_ROOT / "config" / "runs"
    r106, r107 = (json.loads((runs / f"{r}.json").read_text()) for r in ("r106", "r107"))
    assert r107["kernel"] == {**r106["kernel"], "baryon_pair_s_wave": True}
    assert r107["topology"] == r106["topology"]
    for block in ("fit", "diffractive_split", "isovector", "fixed", "statistics"):
        assert r107[block] == r106[block]
    assert [p["name"] for p in r107["parameters"]] == [p["name"] for p in r106["parameters"]]
    assert r107["generation_0"]["run"] == "r106_baryon_tables_2026-10-21"
    assert r107["seeds"]["raw_first"] > r106["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r106_baryon_tables_2026-10-21" / "branch_report_f0.json"
    if branch.is_file():
        report = json.loads(branch.read_text())
        assert r107["generation_0"]["branch_digest"] == report["digest"]
        gen0 = {p["name"]: p["gen0"] for p in r107["parameters"]}
        for name in r107["generation_0"]["from_fit"]:
            assert float(gen0[name]) == float(report["fitted"][name]), name
