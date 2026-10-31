"""Round-4 regression: dynamically injected chain must satisfy quantum
numbers (incl. charged anti-tower), be structurally identical to the
monorepo-helper-built chain (ordered modes, samplers, seeded bitwise
execution), and leave no terminal unstable resonance after cascading."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from na49_mixing.parallel import GenerationConfig, _build_decay_chain
from na49_mixing.event import Event
from na49_mixing.particle import Particle
from na49_mixing.reaction import Reaction
from na49_mixing import pdg_database
from na49_mixing.breit_wigner import BreitWigner
from na49_mixing.decays import (
    WEAK_DECAY_PARENTS,
    DecayMode,
    ParticleDecays,
    build_kaon_decays,
    build_soft_resonance_decays,
    read_decays,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _chain(rng_seed=1):
    cfg = GenerationConfig(gen_file="x", n_events=1, seed=rng_seed,
                           use_soft_decay_factory=True,
                           inject_antibaryon_resonances=True)
    return _build_decay_chain(cfg, np.random.default_rng(rng_seed))


def _helper_chain(rng_seed=1):
    """Chain built through the monorepo helper wrapper's production call.

    Mirrors the uncached call site in ``benchmark_sps_species``: the five
    decay-building dependencies are the real ``na49_mixing.decays``
    objects imported above, never placeholders.  The wrapper delegates to
    ``na49_mixing.parallel._build_decay_chain``, so this returns whatever
    production builds.
    """
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from helpers.benchmarking import benchmark_sps_species as bss
    gen = os.path.join(_REPO_ROOT, "dev", "mixing2006", "na49_mixing",
                       "generators", "michal", "deltas-to-pion",
                       "delta++.gen")
    spec = bss.ResonanceSpec(label="parity", gen_file=gen,
                             use_soft_decay_factory=True)
    return bss._build_decay_chain(
        spec, np.random.default_rng(rng_seed),
        DecayMode, ParticleDecays, build_kaon_decays,
        build_soft_resonance_decays, read_decays)


def _qns_map():
    """Reuse the audited map from the round-1/2 regression module."""
    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)
    import test_physics_fixes_2026_08 as tf
    return dict(tf.TestQuantumNumberConservation.QNS)


_NEG_INC_TOL = 1e-12


def _injected_chain_violations(chain, qns):
    """Audit one injected chain; return the list of violation messages.

    Negative-increment detection runs inside the per-mode loop so a
    non-monotone cumulative table is reported even when the final
    cumulative is 1.0 (the production builder forces the last entry to
    1.0, which previously masked stale-variable checks placed behind the
    final-cumulative condition).
    """
    viol = []
    for pid, dec in sorted(chain.items()):
        if pid not in qns:
            viol.append(f"unknown parent {pid}")
            continue
        qp, bp, sp = qns[pid]
        prev = 0.0
        for pr, m in zip(dec._probabilities, dec._modes):
            inc = pr - prev
            prev = pr
            if inc < -_NEG_INC_TOL:
                viol.append(f"negative prob {pid}->{m.daughters}")
            qs = [qns.get(d) for d in m.daughters]
            if any(q is None for q in qs):
                viol.append(f"unknown daughter {pid}->{m.daughters}")
                continue
            st = sum(x[2] for x in qs)
            if sorted(m.daughters) == [130, 310]:
                st = 0
            if sum(x[0] for x in qs) != qp:
                viol.append(f"Q {pid}->{m.daughters}")
            if sum(x[1] for x in qs) != bp:
                viol.append(f"B {pid}->{m.daughters}")
            if pid not in WEAK_DECAY_PARENTS and st != sp:
                viol.append(f"S {pid}->{m.daughters}")
        if dec._probabilities and abs(dec._probabilities[-1] - 1.0) > 1e-9:
            viol.append(f"{pid}: final cumulative {dec._probabilities[-1]}")
    return viol


def _synthesized_table(cums):
    """ParticleDecays table for parent rho(770)^0 with given cumulatives.

    Every mode is pi+ pi-, which conserves charge, baryon number and
    strangeness for this parent, so quantum-number checks stay silent and
    monotonicity violations stand alone.
    """
    dec = ParticleDecays(113, rng=np.random.default_rng(5))
    for c in cums:
        dec.add_mode(c, DecayMode(2, [211, -211]))
    return dec


def test_injected_chain_conserves_quantum_numbers():
    qns = _qns_map()
    viol = _injected_chain_violations(_chain(), qns)
    assert not viol, "; ".join(viol[:8])
    # Construction now refuses decreasing cumulatives. The auditor still
    # flags a table whose stored probabilities were mutated after load.
    with pytest.raises(ValueError, match="nondecreasing"):
        _synthesized_table([0.60, 0.40, 1.00])
    assert _injected_chain_violations({113: _synthesized_table(
        [0.60, 0.80, 1.00])}, qns) == []
    mutated = _synthesized_table([0.60, 0.80, 1.00])
    mutated._probabilities[1] = 0.40
    msgs = _injected_chain_violations({113: mutated}, qns)
    assert len(msgs) == 1 and "negative prob" in msgs[0], msgs


def _ordered_mode_signature(dec):
    """Ordered per-mode signature; daughters are NOT sorted.

    Each entry is (incremental probability, daughter tuple, per-daughter
    sampler signature), in table order.  A sampler signature is ``None``
    or a ``(sampler type, Breit-Wigner parameter tuple)`` pair, so any
    change of one increment, one daughter ordering, or one sampler
    parameter changes the signature.
    """
    sig = []
    prev = 0.0
    for pr, m in zip(dec._probabilities, dec._modes):
        samplers = tuple(
            (type(bw), tuple(bw.params)) if bw is not None else None
            for bw in m.bws)
        sig.append((pr - prev, tuple(m.daughters), samplers))
        prev = pr
    return sig


def test_helper_wrapper_chain_is_structurally_identical_to_package_chain():
    a = _chain(rng_seed=3)
    b = _helper_chain(rng_seed=3)
    assert a is not b
    assert set(a) == set(b)
    for pid in sorted(a):
        sa = _ordered_mode_signature(a[pid])
        sb = _ordered_mode_signature(b[pid])
        assert sa == sb, f"ordered mode sequence mismatch at parent {pid}"

    # Comparator strictness on live data: perturbing a single incremental
    # probability or a single Breit-Wigner parameter must be detected.
    sig = _ordered_mode_signature(b[-9912201])
    bumped = [(x + (1e-12 if i == 0 else 0.0), d, s)
              for i, (x, d, s) in enumerate(sig)]
    assert bumped != sig
    mi, bi = next((i, k) for i, (_, _, ss) in enumerate(sig)
                  for k, s in enumerate(ss) if s is not None)
    t0, p0 = sig[mi][2][bi]
    rows = [list(row) for row in sig]
    rows[mi][2] = tuple(
        (t0, p0[:1] + (p0[1] + 1e-9,) + p0[2:]) if k == bi else s
        for k, s in enumerate(rows[mi][2]))
    assert tuple(map(tuple, rows)) != sig


# Deterministic execution parity: both chains must produce bit-identical
# final states under identical seeds.  Parent masses mirror the effective
# tower level defaults (production mass_type=3 sampling peaks).  Several
# seeds per parent widen channel coverage of the comparison.

_EXEC_SEED = 41
_EVENT_SEEDS = tuple(range(58, 66))
_EXEC_PARENT_MASSES = {
    -212214: 1.72,   # anti-N(1720)- conjugate partner
    -9912201: 1.90,  # anti-N_eff(1900)- charged partner
    -9922201: 2.10,  # anti-N_eff(2100)- charged partner
    -9932201: 2.30,  # anti-N_eff(2300)- charged partner
    -9911201: 1.90,  # anti-N_eff(1900)0 neutral tower
}


def _decay_tracks(chain, pid, event_seed):
    """Decay one fixed parent through *chain*; return bit-exact tracks."""
    ev = Event(rng=np.random.default_rng(event_seed))
    ev._decay_chain = chain
    parent = Particle.from_id(0.3, -0.2, 25.0, pid, Reaction(158.0))
    parent.set_mass(_EXEC_PARENT_MASSES[pid])
    ev.add_particle(parent)
    ev.mark_primary(parent)
    ev.decay_cascade(max_passes=8)
    stats = ev.decay_stats()
    assert stats["n_primary_decayed"] == 1, f"{pid}: primary not decayed"
    assert stats["n_decay_kinematic_failures"] == 0, (
        f"{pid}: kinematic failures ({stats})")
    # float.hex() makes the comparison bit-exact (it distinguishes
    # signed zeros and NaN payloads that == would collapse).
    return [(p.get_id(), p.get_px().hex(), p.get_py().hex(),
             p.get_pz().hex()) for p in ev.particles()]


def test_chains_execute_bitwise_identically():
    chain_pkg = _chain(rng_seed=_EXEC_SEED)
    chain_hlp = _helper_chain(rng_seed=_EXEC_SEED)
    for pid in sorted(_EXEC_PARENT_MASSES):
        for event_seed in _EVENT_SEEDS:
            tracks_pkg = _decay_tracks(chain_pkg, pid, event_seed)
            tracks_hlp = _decay_tracks(chain_hlp, pid, event_seed)
            assert len(tracks_pkg) >= 2, f"{pid}: no daughter tracks"
            assert tracks_pkg == tracks_hlp, (
                f"execution divergence at {pid}, seed {event_seed}")


# All 13 dynamically injected anti-baryon parents: every injected state
# must cascade to a fully stable final state, never leave a terminal
# unstable resonance, and never exhaust the kinematic retry budget.
_ALL_INJECTED_PARENTS = (
    -12112, -2116, -22112, -102112, -122112, -212114,
    -9911201, -9921201, -9931201,
    -212214, -9912201, -9922201, -9932201,
)

_N_TRIALS_PER_STATE = 500

# Lightest strong two-body channel of the injected parents is N + pion
# (threshold 1.079 GeV); keeping every sampled parent mass above this edge
# guarantees an open channel exists, so any failure below reflects real
# physics rather than starved kinematics.
_MIN_PARENT_MASS_GEV = 1.12


def _truncated_parent_bw(pdg_id, rng):
    """Truncated p-wave Breit-Wigner from pdg_database mass and width.

    Mirrors production mass_type=3 sampling of the effective-tower cards:
    peak and width come from the PDG table, the lower support edge sits two
    widths below the peak (never below ``_MIN_PARENT_MASS_GEV``), and a
    linear damping window truncates the upper tail from mass+0.6 GeV to
    mass+1.2 GeV, matching generators/benchmark/nstar_eff*.gen and the
    tower rewrite performed by benchmark_sps_species.
    """
    rec = pdg_database.get_particle(int(pdg_id))
    mass = float(rec.mass)
    width = float(rec.width)
    schwel = max(mass - 2.0 * width, _MIN_PARENT_MASS_GEV)
    return BreitWigner(
        schwel, mass, width, 1.0, mass + 0.60, mass + 1.20, rng=rng
    )


def test_no_terminal_unstable_after_cascade():
    chain = _chain(rng_seed=9)
    react = Reaction(158.0)
    problems = []
    n_trials = 0
    for pid in _ALL_INJECTED_PARENTS:
        assert pid in chain, f"{pid} missing from the injected chain"
        bw = _truncated_parent_bw(pid, np.random.default_rng(1000 + abs(pid)))
        masses = bw.get_random_n(_N_TRIALS_PER_STATE)
        for trial, mass in enumerate(masses):
            ev = Event(rng=np.random.default_rng((abs(int(pid)), trial)))
            ev._decay_chain = chain
            parent = Particle.from_id(0.3, -0.2, 25.0, pid, react)
            parent.set_mass(float(mass))
            ev.add_particle(parent)
            ev.mark_primary(parent)
            ev.decay_cascade(max_passes=10)
            stats = ev.decay_stats()
            n_trials += 1
            leftovers = [
                p.get_id() for p in ev.particles()
                if p.get_id() in ev._decay_chain
            ]
            if leftovers:
                problems.append(
                    f"{pid} trial {trial} mass {float(mass):.3f}: "
                    f"terminal unstable {leftovers[:4]}"
                )
            if stats["n_primary_decayed"] != 1:
                problems.append(
                    f"{pid} trial {trial} mass {float(mass):.3f}: "
                    f"primary not decayed ({stats})"
                )
            if stats["n_decay_kinematic_failures"] != 0:
                problems.append(
                    f"{pid} trial {trial} mass {float(mass):.3f}: "
                    f"kinematic failures ({stats})"
                )
    assert n_trials == _N_TRIALS_PER_STATE * len(_ALL_INJECTED_PARENTS)
    assert not problems, "; ".join(problems[:8])


# ---------------------------------------------------------------------------
# Structural charge conjugation of the charged anti-tower partners (P1).
#
# Each charged anti-tower parent must mirror its matter positive-tower
# partner: same ordered incremental probabilities, charge-conjugate
# daughter lists, and identical per-daughter spectral samplers.
# ---------------------------------------------------------------------------

_CHARGED_ANTITOWER_PAIRS = (
    (-212214, 212214),
    (-9912201, 9912201),
    (-9922201, 9922201),
    (-9932201, 9932201),
)

_SELF_CONJUGATE_MESONS = frozenset({111, 113, 221})
_INC_TOL = 1e-12


def _conj_daughters(daughters):
    """Charge conjugate of a daughter list; self-conjugate mesons unchanged."""
    return [d if d in _SELF_CONJUGATE_MESONS else -d for d in daughters]


def _trimmed_increments(dec):
    """Ordered incremental probabilities; trailing FP dust removed.

    The matter factory closes residual mass explicitly, so branch sums that
    land on 1.0 - epsilon can emit two final modes carrying ~0 probability.
    Increments at or below ``_INC_TOL`` at the end of the table carry no
    physics and would otherwise break exact structural comparisons.
    """
    incs = []
    prev = 0.0
    for pr in dec._probabilities:
        incs.append(pr - prev)
        prev = pr
    while incs and incs[-1] <= _INC_TOL:
        incs.pop()
    return incs


def _leading_block(dec):
    """(increment, mode) pairs up to and including the Lambda-K channel.

    The leading hadronic block (Npi, Deltapi, Neta, Nrho, Lambda K) is the
    region where matter tower and charged anti-tower are structural charge
    conjugates mode by mode; the tail differs by design because anti towers
    are terminal (their cascade fraction folds into the Nbar-pi closure).
    """
    out = []
    prev = 0.0
    for pr, m in zip(dec._probabilities, dec._modes):
        out.append((pr - prev, m))
        prev = pr
        if any(abs(d) == 3122 for d in m.daughters):
            break
    return out


def _assert_leading_block_conjugate(m_anti, m_matter):
    block_a = _leading_block(m_anti)
    block_h = _leading_block(m_matter)
    assert len(block_a) == len(block_h), (
        f"leading-block length mismatch: {len(block_a)} vs {len(block_h)}")
    for i, ((ia, ma), (ih, mh)) in enumerate(zip(block_a, block_h)):
        assert ia == pytest.approx(ih, rel=1e-12, abs=1e-15), (
            f"increment {i}: {ia} vs {ih}")
        expected = _conj_daughters(mh.daughters)
        assert list(ma.daughters) == expected, (
            f"daughter conjugation mismatch in mode {i}: "
            f"{list(ma.daughters)} vs {expected}")
        for bw_a, bw_h in zip(ma.bws, mh.bws):
            assert (bw_a is None) == (bw_h is None), (
                f"sampler presence mismatch in mode {i}")
            if bw_a is not None:
                assert type(bw_a) is type(bw_h), f"sampler type mismatch in mode {i}"
                assert tuple(bw_a.params) == tuple(bw_h.params), (
                    f"sampler parameters mismatch in mode {i}")


def _assert_full_increment_parity(m_anti, m_matter):
    a_inc = _trimmed_increments(m_anti)
    h_inc = _trimmed_increments(m_matter)
    assert len(a_inc) == len(h_inc), (
        f"mode count mismatch after trimming: {len(a_inc)} vs {len(h_inc)}")
    for i, (x, y) in enumerate(zip(a_inc, h_inc)):
        assert x == pytest.approx(y, rel=1e-12, abs=1e-15), (
            f"increment {i}: {x} vs {y}")


def _nrho_family_mass(dec):
    """Total probability mass carried by N(Nbar)+rho modes."""
    total, prev = 0.0, 0.0
    for pr, m in zip(dec._probabilities, dec._modes):
        inc = pr - prev
        prev = pr
        if any(d in (113, 213, -213) for d in m.daughters):
            total += inc
    return total


def test_charged_antitower_is_exact_charge_conjugate():
    chain = _chain(rng_seed=11)
    matter_lib = build_soft_resonance_decays(rng=np.random.default_rng(11))
    for anti_pid, matter_pid in _CHARGED_ANTITOWER_PAIRS:
        m_anti = chain[anti_pid]
        m_matter = matter_lib[matter_pid]
        # Identical ordered incremental probabilities under default wiring.
        _assert_full_increment_parity(m_anti, m_matter)
        # Leading hadronic block: exact charge conjugation, same samplers.
        _assert_leading_block_conjugate(m_anti, m_matter)
        # Cumulative probabilities never decrease (zero-width FP-dust
        # closure modes from the factory are tolerated) and no real
        # increment is negative; the last entry is forced to 1.0.
        for dec in (m_anti, m_matter):
            cums = list(dec._probabilities)
            assert all(b >= a for a, b in zip(cums, cums[1:])), (
                f"{dec} cumulative probabilities not monotone")
            incs = [cums[0]] + [b - a for a, b in zip(cums, cums[1:])]
            assert all(inc > -_INC_TOL for inc in incs), (
                f"{dec} has a negative increment")
            assert cums[-1] == pytest.approx(1.0)


def test_charged_antitower_override_response_matches_matter():
    ovr_key, ovr_val = "nstar_eff_l1_br_nrho", 0.10
    cfg = GenerationConfig(gen_file="x", n_events=1, seed=13,
                           use_soft_decay_factory=True,
                           inject_antibaryon_resonances=True,
                           soft_decay_overrides=((ovr_key, ovr_val),))
    chain = _build_decay_chain(cfg, np.random.default_rng(13))
    matter_lib = build_soft_resonance_decays(
        rng=np.random.default_rng(13), branching_overrides={ovr_key: ovr_val})
    for anti_pid, matter_pid in _CHARGED_ANTITOWER_PAIRS:
        # Shared structural channels respond identically to the override.
        _assert_leading_block_conjugate(chain[anti_pid], matter_lib[matter_pid])
        fa = _nrho_family_mass(chain[anti_pid])
        fh = _nrho_family_mass(matter_lib[matter_pid])
        assert fa == pytest.approx(fh), (
            f"override response differs between {anti_pid} and {matter_pid}")
    # The override actually took effect on both level-1 partners.
    assert _nrho_family_mass(chain[-9912201]) == pytest.approx(ovr_val)
    assert _nrho_family_mass(matter_lib[9912201]) == pytest.approx(ovr_val)

    # Over-unity regime: shared level-1 keys pushed above unity must retune
    # matter and charged anti towers by proportional renormalization (the
    # same policy as decays.py::_add_nstar_extended and _add_antinstar),
    # never by truncating channels at the cumulative guard.  With
    # br_nstarpi pinned to zero the matter partner normalizes the same
    # five-family set as the charged tower, so exact conjugate parity holds.
    ovr_over = (
        ("nstar_eff_l1_br_npi", 0.15),
        ("nstar_eff_l1_br_deltapi", 0.4375),
        ("nstar_eff_l1_br_nrho", 0.35),
        ("nstar_eff_l1_br_neta", 0.0625),
        ("nstar_eff_l1_br_lambdak", 0.0625),
        ("nstar_eff_l1_br_nstarpi", 0.0),
    )  # the five shared families sum to 1.0625 > 1
    cfg_over = GenerationConfig(
        gen_file="x", n_events=1, seed=17,
        use_soft_decay_factory=True,
        inject_antibaryon_resonances=True,
        soft_decay_overrides=ovr_over)
    chain_over = _build_decay_chain(cfg_over, np.random.default_rng(17))
    matter_lib_over = build_soft_resonance_decays(
        rng=np.random.default_rng(17), branching_overrides=dict(ovr_over))
    anti_over = chain_over[-9912201]
    matter_over = matter_lib_over[9912201]
    _assert_full_increment_parity(anti_over, matter_over)
    _assert_leading_block_conjugate(anti_over, matter_over)
    # All nine renormalized modes survive on both towers, including the
    # Lambda-K channel that the old cumulative-guard truncation dropped.
    assert len(_trimmed_increments(anti_over)) == 9
    assert len(_trimmed_increments(matter_over)) == 9
    assert any(abs(d) == 321 for m in anti_over._modes for d in m.daughters)
    assert any(abs(d) == 321 for m in matter_over._modes for d in m.daughters)
    for dec in (anti_over, matter_over):
        cums = list(dec._probabilities)
        assert all(b >= a for a, b in zip(cums, cums[1:])), (
            f"{dec} cumulative probabilities not monotone")
        assert cums[-1] == pytest.approx(1.0)
    # Proportional response: each nrho family carries br_nrho / total.
    expected_nrho = pytest.approx(0.35 / 1.0625, rel=1e-12, abs=1e-15)
    assert _nrho_family_mass(anti_over) == expected_nrho
    assert _nrho_family_mass(matter_over) == expected_nrho

    # General mixed regime: with br_nstarpi left at its default the matter
    # partner normalizes its extra feed-down family as well (matter total
    # 1.2125 versus the charged tower's 1.0625), so per-family fractions
    # differ by design between the two towers.  The charged tower must
    # still keep every one of its channels instead of truncating.
    ovr_mixed = tuple(
        (k, v) for k, v in ovr_over if k != "nstar_eff_l1_br_nstarpi")
    cfg_mixed = GenerationConfig(
        gen_file="x", n_events=1, seed=19,
        use_soft_decay_factory=True,
        inject_antibaryon_resonances=True,
        soft_decay_overrides=ovr_mixed)
    chain_mixed = _build_decay_chain(cfg_mixed, np.random.default_rng(19))
    matter_lib_mixed = build_soft_resonance_decays(
        rng=np.random.default_rng(19), branching_overrides=dict(ovr_mixed))
    anti_mixed = chain_mixed[-9912201]
    matter_mixed = matter_lib_mixed[9912201]
    assert len(_trimmed_increments(anti_mixed)) == 9
    assert any(abs(d) == 321 for m in anti_mixed._modes for d in m.daughters)
    assert len(_trimmed_increments(matter_mixed)) == 11
    assert any(abs(d) == 321 for m in matter_mixed._modes for d in m.daughters)
    for dec in (anti_mixed, matter_mixed):
        cums = list(dec._probabilities)
        assert all(b >= a for a, b in zip(cums, cums[1:])), (
            f"{dec} cumulative probabilities not monotone")
        assert cums[-1] == pytest.approx(1.0)
    assert _nrho_family_mass(anti_mixed) == pytest.approx(
        0.35 / 1.0625, rel=1e-12, abs=1e-15)
    assert _nrho_family_mass(matter_mixed) == pytest.approx(
        0.35 / 1.2125, rel=1e-12, abs=1e-15)


def _nstarmeson_family_mass(dec, meson_tower_id):
    """Total probability mass of the N*->N*'+M* link family (marker M*)."""
    total, prev = 0.0, 0.0
    for pr, m in zip(dec._probabilities, dec._modes):
        inc = pr - prev
        prev = pr
        if meson_tower_id in m.daughters:
            total += inc
    return total


def _neta_family_mass(dec):
    """Total probability mass of the (N)bar-eta family (marker eta 221)."""
    return _nstarmeson_family_mass(dec, 221)


def test_charged_antitower_nstarmeson_over_unity_renormalizes_like_other_towers():
    """REV-5b2f37d-02: an out-of-band br_nstarmeson override (above the D4
    base_k band, without a compensating br_nstarpi reduction) renormalizes
    proportionally on the charged anti tower exactly as the over-unity
    policy already does on the matter tower (decays.py::_add_nstar_extended)
    and the neutral anti tower (_add_antinstar).  Before the fix the
    charged link was silently truncated at the cumulative guard while the
    other two towers rescaled.

    Level-2 arithmetic: the five shared families default to 0.08 + 0.30 +
    0.22 + 0.05 + 0.05 = 0.70, matter and neutral-anti additionally carry
    br_nstarpi = 0.30, and br_nstarmeson = 0.60, so the tower totals are
    1.60 (matter, neutral anti) and 1.30 (charged anti).  Every family
    keeps declared/total on its own tower; the totals differ between the
    matter and charged towers by construction because the charged tower
    carries no br_nstarpi branch (documented residual asymmetry).
    """
    ovr_key, ovr_val = "nstar_eff_l2_br_nstarmeson", 0.60
    cfg = GenerationConfig(gen_file="x", n_events=1, seed=23,
                           use_soft_decay_factory=True,
                           inject_antibaryon_resonances=True,
                           soft_decay_overrides=((ovr_key, ovr_val),))
    chain = _build_decay_chain(cfg, np.random.default_rng(23))
    matter_lib = build_soft_resonance_decays(
        rng=np.random.default_rng(23), branching_overrides={ovr_key: ovr_val})

    m2_meson = 9900213  # level-2 M* tower daughter of every nstarmeson link
    towers = (
        # (name, decay table, link daughters, link share, Nbar-eta share)
        ("matter", matter_lib[9922201], [9912201, 9900213],
         0.60 / 1.60, 0.05 / 1.60),
        ("neutral_anti", chain[-9921201], [-9911201, 9900213],
         0.60 / 1.60, 0.05 / 1.60),
        ("charged_anti", chain[-9922201], [-9912201, 9900213],
         0.60 / 1.30, 0.05 / 1.30),
    )
    for name, dec, link_daughters, link_expected, neta_expected in towers:
        # Exactly one link mode, carrying the charge-conjugate daughters.
        link_modes = [m for m in dec._modes if m2_meson in m.daughters]
        assert len(link_modes) == 1, name
        assert list(link_modes[0].daughters) == link_daughters, name
        link = _nstarmeson_family_mass(dec, m2_meson)
        assert link == pytest.approx(link_expected, rel=1e-12, abs=1e-15), (
            f"{name}: nstarmeson link {link} != renormalized {link_expected}")
        # The N(Nbar)-eta family survives proportionally too; under the old
        # cumulative-guard truncation it would have kept its unscaled value.
        neta = _neta_family_mass(dec)
        assert neta == pytest.approx(neta_expected, rel=1e-12, abs=1e-15), (
            f"{name}: N-eta family {neta} != renormalized {neta_expected}")
        cums = list(dec._probabilities)
        assert all(b >= a for a, b in zip(cums, cums[1:])), (
            f"{name}: cumulative probabilities not monotone")
        assert cums[-1] == pytest.approx(1.0), name
