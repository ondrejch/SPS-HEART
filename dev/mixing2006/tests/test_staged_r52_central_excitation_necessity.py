"""Hermetic tests for the R52 run driver (the central-excitation
necessity test).

Task ``TASK-20260925-r5-run-r52-central-excitation-necessity`` (plan
section 11 run R52, lines 920-922): the pre-registered zero-model seed
map (8 seeds, 20300953-20300960) and the event-index partition
(1_200_000_000 + i, i < 100,000); the ND-zero and renormalization
algebra on a pinned R51 fixture; the held-out set invariants (the
central species, the |y| < 0.5 row window, the multiplicity tail cut,
and the disjointness from the R51 training observables); the one-model
reuse guard (no regeneration on the one model; the bit-identity
machinery catches a tampered sealed record); the retention-bound
algebra on a pinned likelihood fixture; the R53 trigger algebra; the
smoke-mode refusal conventions; the layout sync; and the freeze re-pin
register.  No Monte Carlo generation at run scale in the deterministic
legs (the sealed-dir pins read the sealed run dirs when the artifact
store is present).
"""

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from tests._freeze_inventory import inventory_count as _inventory_count


#: The sealed driver's per-event mass-shell check divides the deviation by
#: M^2, and the kernel's radiative channels (2026-10-02,
#: TASK-20261002-kernel-feedover-threshold-radiative) put massless photons
#: in the final state, so the driver's hermetic generation cannot pass on
#: the current kernel.  The sealed archive tests stay active; the current
#: pipeline is the R71 register's.
_SEALED_KERNEL_SKIP = pytest.mark.skip(
    reason="the sealed driver predates the radiative channels (massless photons) of the current kernel")


def _load_driver():
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "helpers" / "benchmarking"
        / "staged_r52_central_excitation_necessity.py",
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location(
                "staged_r52_central_excitation_necessity", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        "the R52 driver was not found next to the test file")


r52 = _load_driver()


# ---------------------------------------------------------------------------
# The run identity constants (the card's pre-registered identity)
# ---------------------------------------------------------------------------
def test_run_identity_constants():
    """The run identity: R52, the card, the execution-date run name (the
    card moves the date with the actual execution; the pre-registered
    2026-09-25 identity re-pins to the execution date), the T2
    statistics, the zero-model seed block, and the smoke block."""
    assert r52.RUN_ID == "R52"
    assert r52.CARD == "TASK-20260925-r5-run-r52-central-excitation-necessity"
    assert r52.RUN_NAME.startswith("r52_central_excitation_necessity_")
    assert r52.DEFAULT_RUN_DIR == f"dev/00runs/{r52.RUN_NAME}"
    assert r52.MASTER_SEED == 20260925
    assert r52.N_REPLICAS == 8
    assert r52.EVENTS_PER_REPLICA == 100_000
    assert r52.N_EVENTS_TOTAL == 800_000
    assert r52.REPLICA_SEEDS == tuple(20300953 + k for k in range(8))
    assert r52.EVENT_INDEX_BASE == 1_200_000_000
    assert r52.EVENT_INDEX_RULE == "1_200_000_000 + i, i < 100,000 (per replica)"
    assert r52.SMOKE_SEED == 20260925 + 40_000 + 12_000 == 20312925
    assert r52.SMOKE_EVENTS == 2_000
    assert r52.SMOKE_EVENT_INDEX_BASE == 1_600_000_000
    assert r52.PROBE_REPLICA == 0
    assert r52.MULTIPLICITY_BINS == tuple(range(2, 15))
    assert r52.CLASS_ORDER == ("sd_projectile", "sd_target", "dd", "nd")


def test_seed_map_and_partition():
    """The seed map (index item 9): the 8 raw zero-model seeds in order,
    the block disjoint from every previously used raw block (including
    the sealed R50 and R51 blocks) and from every reserved Round 5
    block, the event-index partition disjoint from every recorded
    partition, and the smoke block disjoint from the production block
    by seed and by partition."""
    seeds = [r52.replica_seed(k) for k in range(8)]
    assert seeds == list(range(20300953, 20300961))
    assert seeds == list(r52.REPLICA_SEEDS)
    report = r52.seed_disjointness_report()
    assert report["disjoint"] is True
    assert report["shared_raw_seeds"] == {}
    r52_lo, r52_hi = report["r52_event_index_range"]
    assert r52_lo == 1_200_000_000 and r52_hi == 1_200_100_000
    for name, (lo, hi) in r52.RESERVED_ROUND5_PARTITIONS.items():
        if name in ("r50_used", "r51_used"):
            continue
        assert r52_hi <= lo or hi <= r52_lo, name
    for name, block in r52.PREVIOUS_USED_BLOCKS.items():
        p_range = block.get("event_index_range")
        if p_range is not None:
            assert r52_hi <= p_range[0] or p_range[1] <= r52_lo, name
    for name, (lo, hi) in r52.RESERVED_ROUND5_BLOCKS.items():
        assert not (set(range(lo, hi + 1)) & set(r52.REPLICA_SEEDS)), name
    assert r52.SMOKE_SEED not in set(r52.REPLICA_SEEDS)
    assert r52.SMOKE_EVENT_INDEX_BASE >= r52_hi
    # the one-side reuse re-consumes the sealed R51 block by design
    assert "20300945-20300952" in report["one_side_reuse"]


# ---------------------------------------------------------------------------
# The ND-zero and renormalization algebra (pinned R51 fixture)
# ---------------------------------------------------------------------------
def test_frozen_values_constants():
    """The frozen R51 fitted point (byte-exact constants) and the
    renormalization algebra: the fraction closure, the ND fraction set
    to exactly 0.0, the renormalized sum to one, and the frozen
    epsilon/B_SD inside the R51 pre-registered brackets."""
    assert r52.FROZEN_FIT == {
        "sd_projectile_fraction": 3.619686246956965e-09,
        "sd_target_fraction": 2.921678336937106e-07,
        "dd_fraction": 0.9999997010778905,
        "nd_fraction": 3.1345895878942542e-09,
        "epsilon": 0.0033443027832747944,
        "b_sd": 1.8457918026393214,
    }
    status = r52.assert_frozen_values()
    assert status["pass"] is True
    renorm = status["renormalized_fractions"]
    assert renorm["nd"] == 0.0
    assert abs(math.fsum(renorm.values()) - 1.0) <= 1e-12
    norm = 1.0 - r52.FROZEN_FIT["nd_fraction"]
    assert renorm["sd_projectile"] == \
        r52.FROZEN_FIT["sd_projectile_fraction"] / norm
    assert renorm["sd_target"] == \
        r52.FROZEN_FIT["sd_target_fraction"] / norm
    assert renorm["dd"] == r52.FROZEN_FIT["dd_fraction"] / norm
    params = r52.zero_model_params()
    assert params.class_fractions["nd"] == 0.0
    assert params.epsilon == r52.FROZEN_FIT["epsilon"]
    assert params.b_sd == r52.FROZEN_FIT["b_sd"]
    assert params.p_n2 == 0.0
    assert params.remnant_weights == dict(r52.NOMINAL_PARAMS.remnant_weights)


def test_renormalization_algebra_on_pinned_fixture(monkeypatch):
    """The renormalization algebra on a pinned R51-style fixture (the
    R50 nominal point: f_nd = 0.25 renormalizes the other three to 1/3
    each) and the refusal on a degenerate fixture (f_nd = 1 leaves an
    empty renormalized mass)."""
    fixture = {
        "sd_projectile_fraction": 0.25,
        "sd_target_fraction": 0.25,
        "dd_fraction": 0.25,
        "nd_fraction": 0.25,
        "epsilon": 0.0,
        "b_sd": 8.504140762083612,
    }
    monkeypatch.setattr(r52, "FROZEN_FIT", dict(fixture))
    status = r52.assert_frozen_values()
    renorm = status["renormalized_fractions"]
    assert renorm["nd"] == 0.0
    for name in ("sd_projectile", "sd_target", "dd"):
        assert abs(renorm[name] - 1.0 / 3.0) < 1e-15
    assert abs(math.fsum(renorm.values()) - 1.0) <= 1e-12
    degenerate = dict(fixture, nd_fraction=1.0, dd_fraction=0.25)
    degenerate["dd_fraction"] = 0.25
    monkeypatch.setattr(r52, "FROZEN_FIT", degenerate)
    with pytest.raises(SystemExit):
        r52.assert_frozen_values()


# ---------------------------------------------------------------------------
# The held-out set invariants
# ---------------------------------------------------------------------------
def test_heldout_set_constants():
    """The held-out set invariants (index item 4): the central species
    (the four published plus the model-only pi0), the |y| < 0.5 window,
    the multiplicity tail quantile, the retention bound at 2.0, the
    species particle ids and masses, and the pi0 geometry species."""
    assert r52.PUBLISHED_SPECIES == ("pi_plus", "pi_minus", "k_plus",
                                     "k_minus")
    assert r52.PI0_SPECIES == "pi0"
    assert r52.PI0_GEOMETRY_SPECIES == "pi_minus"
    assert r52.PI0_ROW_MASS_GEV == 0.1349766
    assert r52.CENTRAL_WINDOW == 0.5
    assert r52.TAIL_QUANTILE == 0.90
    assert r52.RETENTION_SE_BOUND == 2.0
    assert r52.SPECIES_PDGS == {"pi_plus": 211, "pi_minus": -211,
                                "pi0": 111, "k_plus": 321, "k_minus": -321}
    assert r52.SPECIES_MASSES["pi_plus"] == 0.1395699
    assert r52.SPECIES_MASSES["k_plus"] == 0.493677
    assert r52.EVENT_CLOSURE_TOL == 1e-9


def test_row_y_cm_window_algebra():
    """The row-window algebra: y = 0 at x_F = 0, the rapidity is odd in
    x_F, and the |y| < 0.5 selection on a synthetic row set picks the
    near-central rows only (a boundary row just outside the window is
    excluded, checked never bled)."""
    mass = 0.1395699
    sqrt_s = 17.270205201372406
    assert r52._row_y_cm(0.0, 0.3, mass, sqrt_s) == 0.0
    y_pos = r52._row_y_cm(0.01, 0.3, mass, sqrt_s)
    y_neg = r52._row_y_cm(-0.01, 0.3, mass, sqrt_s)
    assert abs(y_pos + y_neg) < 1e-12
    assert 0.0 < abs(y_pos) < 0.5
    y_far = r52._row_y_cm(0.075, 0.05, mass, sqrt_s)
    assert abs(y_far) > 0.5


def test_heldout_geometry_central_rows():
    """The central published rows of the sealed R01 tables (when the
    artifact store is present): the row-mid |y^CM| < 0.5 selection and
    the pi0 geometry on the pi- table."""
    r01_dir = r52.MONO_ROOT / r52.R01_ARCHIVE
    if not r01_dir.is_dir():
        pytest.skip("the sealed R01 archive is absent on this checkout")
    _raw_config, tables = r52.r51._load_production_tables()
    sealed = r52.r51._load_sealed_r01(tables)
    geometry = r52.r51.build_geometry(tables)
    sqrt_s_data = r52.r51._sqrt_s_conversion()
    held = r52.build_heldout_geometry(tables, sealed, geometry, sqrt_s_data)
    counts = {species: held["species"][species]["central_rows"]
              for species in r52.PUBLISHED_SPECIES}
    assert counts["pi_plus"] == 58 and counts["pi_minus"] == 58
    assert counts["k_plus"] == 42 and counts["k_minus"] == 42
    for species in r52.PUBLISHED_SPECIES:
        block = held["species"][species]
        assert block["central_pick"].size == block["central_rows"]
        assert np.all(np.abs(block["y_mid_central"]) < r52.CENTRAL_WINDOW)
        assert np.all(block["d_central"] == np.asarray(
            sealed[species]["d"], dtype=float)[
                np.abs(np.asarray([
                    r52._row_y_cm(x, p, block["mass_gev"], sqrt_s_data)
                    for x, p in zip(sealed[species]["xF_mid"],
                                    sealed[species]["pT_mid"])]))
                < r52.CENTRAL_WINDOW])
    pi0 = held["pi0"]
    assert pi0["geometry_species"] == "pi_minus"
    assert pi0["data_side"] is None
    assert pi0["central_pick"].size == pi0["central_rows"]
    assert pi0["central_rows"] >= 1
    # the pi0 row window sits at the pi0 mass, not the pion mass
    assert pi0["mass_gev"] == 0.1349766


def test_heldout_disjointness_from_r51_training():
    """The held-out set's disjointness from the R51 training
    observables (plan section 7.4): the central pi/K spectra never
    appear in the R51 training statistic set (the leading-proton
    spectrum, the B_transport spectrum, the gap fraction, and the
    charged-multiplicity distribution); the multiplicity-tail overlap
    is recorded explicitly in the driver's tail note."""
    r51_training_keys = {
        "leading_proton_spectrum", "b_transport_spectrum",
        "gap_fraction_abs_y_lt_0.5", "charged_multiplicity_distribution",
    }
    for species in r52.PUBLISHED_SPECIES:
        family = f"central_spectrum_{species}"
        assert family not in r51_training_keys
    note = r52.MULTIPLICITY_TAIL_NOTE
    assert "subset of the R51 training bins" in note
    assert "recorded explicitly" in note


# ---------------------------------------------------------------------------
# The multiplicity tail cut
# ---------------------------------------------------------------------------
def test_tail_cut_algebra_on_pinned_fixture():
    """The tail-cut algebra on pinned fixtures: a distribution whose
    cumulative reaches 0.90 exactly at a bin, a uniform distribution
    (the threshold inside the support), and the fallback to the support
    top when the quantile is never reached."""
    bins = list(range(2, 15))
    fixture_exact = {
        "gap_fraction": {"value": 0.0, "variance": 1e-12},
        "charged_multiplicity": {
            "bins": bins,
            "distribution": {str(n): (0.9 if n == 6 else 0.1 / 12)
                             for n in bins},
            "variance": {str(n): 1e-12 for n in bins},
        },
    }
    cut = r52.multiplicity_tail_cut(fixture_exact)
    assert cut["threshold_n_ch"] == 6
    assert cut["tail_bins"] == list(range(6, 15))
    fixture_uniform = {
        "gap_fraction": {"value": 0.0, "variance": 1e-12},
        "charged_multiplicity": {
            "bins": bins,
            "distribution": {str(n): 1.0 / 13 for n in bins},
            "variance": {str(n): 1e-12 for n in bins},
        },
    }
    cut = r52.multiplicity_tail_cut(fixture_uniform)
    assert cut["threshold_n_ch"] == 13
    assert cut["tail_bins"] == [13, 14]
    fixture_flat = {
        "gap_fraction": {"value": 0.0, "variance": 1e-12},
        "charged_multiplicity": {
            "bins": bins,
            "distribution": {str(n): 0.0 for n in bins},
            "variance": {str(n): 1e-12 for n in bins},
        },
    }
    cut = r52.multiplicity_tail_cut(fixture_flat)
    assert cut["threshold_n_ch"] == 14
    assert cut["tail_bins"] == [14]


def test_tail_cut_against_sealed_r51_training_distribution():
    """The tail cut on the R51 training distribution (the sealed
    R50-pinned reference the R51 training multiplicity statistic used),
    when the artifact store is present: the cumulative at the threshold
    reaches 0.90 and the threshold is minimal."""
    r50_dir = r52.MONO_ROOT / r52.R50_RUN_DIR
    if not r50_dir.is_dir():
        pytest.skip("the sealed R50 archive is absent on this checkout")
    reference = r52.r51.r50_reference_payload()
    cut = r52.multiplicity_tail_cut(reference)
    dist = reference["charged_multiplicity"]["distribution"]
    threshold = cut["threshold_n_ch"]
    cumulative = math.fsum(
        float(dist[str(n)]) for n in range(2, threshold + 1))
    assert cumulative >= r52.TAIL_QUANTILE
    below = math.fsum(
        float(dist[str(n)]) for n in range(2, threshold))
    assert below < r52.TAIL_QUANTILE
    assert threshold == 6
    assert cut["tail_bins"] == list(range(6, 15))
    assert "subset of the R51 training multiplicity bins" in cut["training_overlap_note"]


# ---------------------------------------------------------------------------
# The retention-bound algebra (pinned likelihood fixture)
# ---------------------------------------------------------------------------
def test_bin_comparison_algebra_on_pinned_fixture():
    """The per-bin comparison algebra on a pinned likelihood fixture:
    the deviances with V = V_data + V_mc, the pooled SE of the deviance
    difference propagated from both models' MC errors, and the
    beyond-bound flag at the 2.0 pooled-SE bound."""
    # d = 1.0; the one model at the data (zero pull); the zero model
    # 0.1 away with tiny MC errors: the bin degrades far beyond 2.0
    entry = r52.bin_comparison(
        d=1.0, v_data=1e-4, m_one=1.0, m_zero=1.1,
        se_one=1e-3, se_zero=1e-3)
    sigma = math.sqrt(1e-4 + 1e-6)
    assert abs(entry["zero_pull"] + 0.1 / sigma) < 1e-12
    assert entry["one_deviance"] == 0.0
    assert abs(entry["zero_deviance"] - (0.1 / sigma) ** 2) < 1e-9
    assert entry["difference"] > 0.0
    deriv = 2.0 * 0.1 / (1e-4 + 1e-6) * 1e-3
    assert abs(entry["pooled_se"] - deriv) < 1e-9
    assert entry["beyond_2x_se"] is True
    assert abs(entry["residual_shift"] - 0.1) < 1e-12
    # the identical models: no degradation, no beyond flag
    entry_same = r52.bin_comparison(
        d=1.0, v_data=1e-4, m_one=1.05, m_zero=1.05,
        se_one=1e-3, se_zero=1e-3)
    assert entry_same["difference"] == 0.0
    assert entry_same["beyond_2x_se"] is False
    # the zero model closer to the data: the difference is negative and
    # the flag stays false (only degradations count)
    entry_better = r52.bin_comparison(
        d=1.0, v_data=1e-4, m_one=1.1, m_zero=1.0,
        se_one=1e-3, se_zero=1e-3)
    assert entry_better["difference"] < 0.0
    assert entry_better["beyond_2x_se"] is False


def test_family_verdict_on_pinned_fixture():
    """The family verdict algebra on pinned fixtures: the degradation at
    the 2.0 pooled-SE bound, the preservation bound on the one model's
    largest pull, and the AND of the two conditions."""
    degrading = r52.bin_comparison(
        d=1.0, v_data=1e-4, m_one=1.0, m_zero=1.1,
        se_one=1e-3, se_zero=1e-3)
    # the one model preserves (zero pull) and the zero model degrades:
    # retained
    verdict = r52.family_verdict([degrading])
    assert verdict["zero_degrades_at_2x_pooled_se"] is True
    assert verdict["max_degradation_pooled_se"] >= 2.0
    assert verdict["one_model_max_abs_pull"] <= 2.0
    assert verdict["one_model_preserves"] is True
    assert verdict["verdict"] == "retained"
    # the one model far from the data: preservation fails, the family is
    # not retained even though the zero model degrades
    far_one = r52.bin_comparison(
        d=1.0, v_data=1e-4, m_one=1.5, m_zero=1.6,
        se_one=1e-3, se_zero=1e-3)
    verdict = r52.family_verdict([far_one])
    assert verdict["one_model_preserves"] is False
    assert verdict["verdict"] == "not retained"
    # no degradation: the family is not retained
    verdict = r52.family_verdict([
        r52.bin_comparison(d=1.0, v_data=1e-4, m_one=1.0, m_zero=1.0,
                           se_one=1e-3, se_zero=1e-3)])
    assert verdict["zero_degrades_at_2x_pooled_se"] is False
    assert verdict["verdict"] == "not retained"
    # the bound is the pre-registered 2.0 and never relaxed
    assert verdict["bound"] == 2.0 == r52.RETENTION_SE_BOUND


def test_r53_trigger_algebra_on_pinned_fixture():
    """The R53 trigger algebra on pinned fixtures: the one model's
    largest tail pull at the 2.0 pooled-SE bound, the zero model's
    residual recorded beside it, and the no-kept-bin case."""
    bad_tail = dict(r52.bin_comparison(
        d=0.023637, v_data=2.31e-7, m_one=0.0538, m_zero=0.0538,
        se_one=2.7e-3, se_zero=2.5e-4), bin=8, d=0.023637, v_data=2.31e-7,
        m_one=0.0538, m_zero=0.0538, se_one=2.7e-3, se_zero=2.5e-4)
    trigger = r52._r53_trigger([dict(bad_tail)], [dict(bad_tail)])
    assert trigger["triggered"] is True
    assert trigger["sigma"] == abs(bad_tail["one_pull"])
    assert trigger["argmax_bin_n_ch"] == 8
    assert trigger["named_residual"] == (
        "multiplicity_tail_residual_one_model_beyond_2x_pooled_se")
    assert trigger["booleans_agree"] is True
    # below the bound: not triggered
    ok_tail = dict(r52.bin_comparison(
        d=0.5, v_data=0.01, m_one=0.505, m_zero=0.505,
        se_one=0.01, se_zero=0.001), bin=6, d=0.5, v_data=0.01,
        m_one=0.505, m_zero=0.505, se_one=0.01, se_zero=0.001)
    trigger = r52._r53_trigger([dict(ok_tail)], [dict(ok_tail)])
    assert trigger["triggered"] is False
    # no kept bin
    trigger = r52._r53_trigger([], [])
    assert trigger["triggered"] is False
    assert trigger["named_residual"] is None


def test_named_residual_register_on_pinned_fixture():
    """The named-residual register assembly on pinned fixtures: every
    beyond-bound bin is named, the weight degeneracy and the inherited
    R51 caveat are named, and the trigger entry lands when fired."""
    degrading = dict(r52.bin_comparison(
        d=1.0, v_data=1e-4, m_one=1.0, m_zero=1.1,
        se_one=1e-3, se_zero=1e-3), bin=3, kept=True)
    families = {"central_spectrum_pi_plus": {"verdict": "not retained"}}
    per_bin = {"central_spectrum_pi_plus": [degrading]}
    weights_one = {"ess_min": 5e-4, "ess_per_block": [5e-4] * 8}
    trigger = {"triggered": True, "named_residual": "trigger_residual",
               "value": 0.03, "pooled_se": 3e-3, "sigma": 10.0,
               "argmax_bin_n_ch": 8}
    named = r52._assemble_named_residuals(
        families, per_bin, weights_one, trigger, r51_inherited=[
            {"kind": "optimizer_non_convergence"}])
    kinds = [entry["kind"] for entry in named]
    assert kinds.count("heldout_bin_degradation_beyond_2x_pooled_se") == 1
    beyond = named[kinds.index("heldout_bin_degradation_beyond_2x_pooled_se")]
    assert beyond["family"] == "central_spectrum_pi_plus"
    assert beyond["bin"] == 3
    assert "weight_degeneracy" in kinds
    assert "r51_inherited_non_convergence" in kinds
    assert "pi0_published_table_absent" in kinds
    assert "multiplicity_tail_training_overlap" in kinds
    assert "trigger_residual" in kinds
    # no degeneracy below the floor: the residual is not named
    weights_ok = {"ess_min": 0.5, "ess_per_block": [0.5] * 8}
    named = r52._assemble_named_residuals(
        families, per_bin, weights_ok,
        {"triggered": False}, r51_inherited=[])
    kinds = [entry["kind"] for entry in named]
    assert "weight_degeneracy" not in kinds
    assert "trigger_residual" not in kinds


# ---------------------------------------------------------------------------
# The one-model reuse guard (no regeneration on the one model)
# ---------------------------------------------------------------------------
def test_one_side_rederivation_bit_identity_and_tamper_guard():
    """The one-side reuse guard: the re-derivation of a small sealed
    R51-style record reproduces every per-event hash and the hash set
    bit-exactly (no new draw), and a tampered sealed record refuses the
    run (the reuse contract's checked-never-repaired guard)."""
    import numpy as np
    catalog = r52._load_catalog()
    fams = r52._load_families()
    seed = 999_001
    base = 1_100_000_000
    n = 3
    records = []
    for i in range(n):
        subs = r52.derive_substreams(seed, 0, base + i)
        rngs = [np.random.default_rng(
            subs.stream_for(label)) for label in (
            "topology_selection", "spectral_state", "exchange_kinematics",
            "decay_channel", "decay_kinematics")]
        label = r52.draw_topology(
            r52.NOMINAL_PARAMS.class_fractions, 0.0, rngs[0])
        event = r52.run_topology(
            label, r52.NOMINAL_PARAMS, catalog, fams,
            rng_topology_selection=rngs[0], rng_spectral_state=rngs[1],
            rng_exchange_kinematics=rngs[2], rng_decay_channel=rngs[3],
            rng_decay_kinematics=rngs[4], p_n2=r52.NOMINAL_PARAMS.p_n2)
        obs = r52.extract_observables(event, r52.SQRT_S)
        conservation = r52.event_conservation(event)
        protons, pbars = r52.spectrum_records(event)
        n_ch = r52.charged_multiplicity(event)
        record = r52.serialize_event(
            event, obs, conservation, protons, pbars, n_ch)
        record["i"] = int(i)
        record["id"] = int(base + i)
        record.pop("h", None)
        record["h"] = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
                "utf-8")).hexdigest()
        records.append(record)
    sealed = {
        "seed": seed,
        "n_events": n,
        "events": records,
        "event_hash_set_sha256": hashlib.sha256(
            "\n".join(r["h"] for r in records).encode("utf-8")).hexdigest(),
    }
    class _Small:
        EVENTS_PER_REPLICA = n
        EVENT_INDEX_BASE = base
    saved_n = r52.EVENTS_PER_REPLICA
    saved_base = r52.r51.EVENT_INDEX_BASE
    try:
        r52.EVENTS_PER_REPLICA = n
        r52.r51.EVENT_INDEX_BASE = base
        lattice, identity = r52.rederive_r51_replica(
            0, sealed, catalog, fams)
        assert identity["bit_identical"] is True
        assert identity["per_event_hash_mismatches"] == 0
        assert identity["event_hash_set_match"] is True
        # the pi/K extraction ran on the identical events
        assert set(lattice) == set(r52.SPECIES_PDGS)
        tampered = json.loads(json.dumps(sealed))
        tampered["events"][1]["h"] = "0" * 64
        with pytest.raises(SystemExit):
            r52.rederive_r51_replica(0, tampered, catalog, fams)
    finally:
        r52.EVENTS_PER_REPLICA = saved_n
        r52.r51.EVENT_INDEX_BASE = saved_base


def test_generate_mode_never_generates_the_one_model():
    """The one-model reuse guard (structural): the driver's generation
    path is the zero-model generator only; the one-side consumption is
    the sealed-archive re-derivation whose docstring records the reuse
    contract, and the driver carries no one-model generation entry
    point."""
    import inspect
    source = inspect.getsource(r52.run_generation)
    assert "generate_zero_replica_events" in source
    assert "never generated here" in source
    source = inspect.getsource(r52.rederive_r51_replica)
    assert "sealed" in source and "bit-identical" in source
    assert "no new draw happens" in source
    # the report's one-model block pins refit False and the reuse record
    assert "refit" in inspect.getsource(r52.run_heldout_evaluation)


def test_zero_generation_refuses_nd_and_elastic_draws(monkeypatch):
    """The zero-model generation refuses a drawn elastic or nd label
    (the ND fraction is exactly 0.0; checked, never repaired)."""
    def _draw(label):
        def inner(fractions, elastic_fraction, rng):
            return label
        return inner
    catalog = r52._load_catalog()
    fams = r52._load_families()
    for label in ("elastic", "nd"):
        monkeypatch.setattr(r52, "draw_topology", _draw(label))
        with pytest.raises(r52.ConservationViolation):
            r52.generate_zero_replica_events(
                20300953, 1, r52.EVENT_INDEX_BASE, catalog, fams)


@_SEALED_KERNEL_SKIP
def test_zero_generation_small_trace():
    """A small real zero-model trace (30 events at the pre-registered
    zero-model parameters): no elastic or nd event, exact (B, Q, S) =
    (2, 2, 0), the four-momentum closure at tolerance, the ``pk``
    records present, and the event-hash set reproducible bit for bit on
    a second generation."""
    catalog = r52._load_catalog()
    fams = r52._load_families()
    records, census, conservation, hash_set = \
        r52.generate_zero_replica_events(
            20300953, 30, r52.EVENT_INDEX_BASE, catalog, fams)
    assert len(records) == 30
    assert set(census) <= {"sd_projectile", "sd_target", "dd"}
    assert "nd" not in census and "elastic" not in census
    assert conservation["violations"] == 0
    assert conservation["worst_closure_relative"] <= r52.EVENT_CLOSURE_TOL
    assert set(conservation["b_q_s_census"]) == {"2,2,0"}
    for record in records:
        assert set(record["pk"]) == set(r52.SPECIES_PDGS)
        assert all(isinstance(v, list) for v in record["pk"].values())
    records2, census2, conservation2, hash_set2 = \
        r52.generate_zero_replica_events(
            20300953, 30, r52.EVENT_INDEX_BASE, catalog, fams)
    assert hash_set2 == hash_set
    assert census2 == census
    assert conservation2 == conservation


def test_zero_replica_record_identity_verification(tmp_path):
    """The zero-replica identity guard: a completed record is reused
    unchanged when its identity matches, and refused on any drift (the
    seed, the mode, the pre-registration digest, the driver digest, or
    a stale event-hash set)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / r52.PREREGISTRATION_NAME).write_text(
        json.dumps({"mode": "production", "zero_model": {
            "nd_fraction": 0.0,
            "renormalized_fractions": dict(r52.RENORMALIZED_FRACTIONS)}}),
        encoding="utf-8")
    record = {
        "schema": r52.REPLICA_SCHEMA,
        "run": "R52",
        "card": r52.CARD,
        "mode": "production",
        "replica": 0,
        "seed": r52.replica_seed(0),
        "n_events": r52.EVENTS_PER_REPLICA,
        "identity": r52._identity_block(run_dir),
        "events": [],
        "event_hash_set_sha256": hashlib.sha256(b"").hexdigest(),
        "class_census": {},
        "conservation": {},
    }
    # the event count mismatch refuses
    with pytest.raises(SystemExit):
        r52._verify_zero_replica_identity(record, 0, run_dir)
    record["n_events"] = r52.EVENTS_PER_REPLICA
    record["events"] = [{"h": "0" * 64}] * r52.EVENTS_PER_REPLICA
    # the stale event-hash set refuses
    with pytest.raises(SystemExit):
        r52._verify_zero_replica_identity(record, 0, run_dir)
    # the seed drift refuses
    record["event_hash_set_sha256"] = hashlib.sha256(
        ("\n".join("0" * 64 for _ in range(r52.EVENTS_PER_REPLICA))
         ).encode("utf-8")).hexdigest()
    with pytest.raises(SystemExit):
        r52._verify_zero_replica_identity(record, 1, run_dir)
    record["replica"] = 1
    # the pre-registration digest drift refuses
    other = tmp_path / "other"
    other.mkdir()
    (other / r52.PREREGISTRATION_NAME).write_text("{}", encoding="utf-8")
    record["identity"] = r52._identity_block(other)
    with pytest.raises(SystemExit):
        r52._verify_zero_replica_identity(record, 1, run_dir)


# ---------------------------------------------------------------------------
# The smoke-mode refusal conventions (the R31/8c and card-19 discipline)
# ---------------------------------------------------------------------------
def test_smoke_mode_refuses_default_run_dir():
    """--smoke refuses the default production run dir (the smoke
    artifacts are never sealed into the production run dir)."""
    default_dir = r52.MONO_ROOT / r52.DEFAULT_RUN_DIR
    with pytest.raises(SystemExit) as exc:
        r52.run_smoke(default_dir)
    assert exc.value.code == 2


def test_smoke_refusal_convention(tmp_path):
    """--gate / --seal / --clean-validate refuse smoke artifacts (the
    R31/8c convention); clean-validate additionally refuses the sealed
    dir (the card-19 seal-ordering discipline) and a second pass."""
    run_dir = tmp_path / "smoke_run"
    run_dir.mkdir()
    (run_dir / r52.PREREGISTRATION_NAME).write_text(
        json.dumps({"mode": "smoke"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        r52._refuse_smoke_artifacts(run_dir)
    run_dir2 = tmp_path / "prod_named"
    run_dir2.mkdir()
    (run_dir2 / r52.PREREGISTRATION_NAME).write_text(
        json.dumps({"mode": "production"}), encoding="utf-8")
    (run_dir2 / "smoke_replica.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        r52._refuse_smoke_artifacts(run_dir2)
    run_dir3 = tmp_path / "no_pre"
    run_dir3.mkdir()
    with pytest.raises(SystemExit):
        r52._refuse_smoke_artifacts(run_dir3)
    # the sealed dir refuses clean-validate
    run_dir4 = tmp_path / "sealed"
    run_dir4.mkdir()
    (run_dir4 / r52.PREREGISTRATION_NAME).write_text(
        json.dumps({"mode": "production"}), encoding="utf-8")
    (run_dir4 / r52.SEAL_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        r52.run_clean_validate(run_dir4)
    # the second clean-validate pass refuses
    run_dir5 = tmp_path / "second_pass"
    run_dir5.mkdir()
    (run_dir5 / r52.PREREGISTRATION_NAME).write_text(
        json.dumps({"mode": "production"}), encoding="utf-8")
    (run_dir5 / r52.REPORT_NAME).write_text("{}", encoding="utf-8")
    (run_dir5 / r52.GATE_RECORD_NAME).write_text("{}", encoding="utf-8")
    (run_dir5 / r52.CLEAN_ENV_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        r52.run_clean_validate(run_dir5)


def test_pre_registration_written_once(tmp_path, monkeypatch):
    """The pre-registration is written once, before any generation; a
    pre-existing pre-registration refuses the run."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / r52.PREREGISTRATION_NAME).write_text(
        "{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        r52.run_pre_registration(run_dir, smoke=True)


def test_no_optimizer_in_driver():
    """R52 carries no optimizer (index item 10): the driver module has
    no scipy.optimize import and no minimize call, the report
    convention fixes optimizer_invoked False and optimizer_uses 0, and
    the run refuses any drifted constant."""
    source = Path(r52.__file__).read_text(encoding="utf-8")
    assert "scipy.optimize" not in source
    assert "minimize(" not in source
    assert r52.RUN_ID == "R52"
    # the acceptance constants
    assert "optimizer_invoked false" in r52.ACCEPTANCE_NOTE


# ---------------------------------------------------------------------------
# The freeze re-pin register
# ---------------------------------------------------------------------------
MANIFEST_REL = "dev/mixing2006/config/v2_singlet/legacy_v1_manifest.json"
MY_TEST_REL = (
    "dev/mixing2006/tests/"
    "test_staged_r52_central_excitation_necessity.py")
SOLVER_CARD = "TASK-20260925-r5-run-r52-central-excitation-necessity"

#: The chain count-pin files re-pinned by this card (the 41 chain files
#: carrying the inventory count pin).
CHAIN_COUNT_PIN_FILES = (
    "dev/mixing2006/tests/test_cascade_ablation.py",
    "dev/mixing2006/tests/test_cascade_alignment_baryonic.py",
    "dev/mixing2006/tests/test_cascade_alignment_mesonic.py",
    "dev/mixing2006/tests/test_cascade_axes.py",
    "dev/mixing2006/tests/test_cascade_form_factor.py",
    "dev/mixing2006/tests/test_cascade_termination.py",
    "dev/mixing2006/tests/test_exchange_ledger_contract_pins.py",
    "dev/mixing2006/tests/test_r4_audit_blend_path_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_baryonic_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_r4_cascade_alignment_mesonic_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_axes_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_decay_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_form_factor_contract_pins.py",
    "dev/mixing2006/tests/test_r4_cascade_termination_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r40_alignment_scan_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r41_nonstrange_pilot_fit_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r42_recoil_decay_separation.py",
    "dev/mixing2006/tests/"
    "test_staged_r42_recoil_decay_separation_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r43_cascade_ablation.py",
    "dev/mixing2006/tests/"
    "test_staged_r43_cascade_ablation_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r44_frozen_nonstrange_validation.py",
    "dev/mixing2006/tests/test_staged_r45_simplified_alignment_freeze.py",
    "dev/mixing2006/tests/"
    "test_staged_r45_simplified_alignment_freeze_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_staged_r46_simplified_recoil_decay_profile.py",
    "dev/mixing2006/tests/"
    "test_staged_r46_simplified_recoil_decay_profile_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r50_pure_topology_templates.py",
    "dev/mixing2006/tests/"
    "test_staged_r50_pure_topology_templates_contract_pins.py",
    "dev/mixing2006/tests/test_staged_r51_topology_fraction_fit.py",
    "dev/mixing2006/tests/"
    "test_staged_r51_topology_fraction_fit_contract_pins.py",
    "dev/mixing2006/tests/test_topology_double_diffraction_contract_pins.py",
    "dev/mixing2006/tests/test_topology_elastic_contract_pins.py",
    "dev/mixing2006/tests/test_topology_mass_laws_contract_pins.py",
    "dev/mixing2006/tests/test_topology_multi_neutral.py",
    "dev/mixing2006/tests/test_topology_multi_neutral_contract_pins.py",
    "dev/mixing2006/tests/test_topology_non_diffractive.py",
    "dev/mixing2006/tests/"
    "test_topology_non_diffractive_contract_pins.py",
    "dev/mixing2006/tests/test_topology_remnants_contract_pins.py",
    "dev/mixing2006/tests/test_topology_runner.py",
    "dev/mixing2006/tests/test_topology_runner_contract_pins.py",
    "dev/mixing2006/tests/"
    "test_topology_single_diffraction_contract_pins.py",
)

#: This card's freeze-exception file list (path-sorted): the manifest,
#: the dual-tree driver, this test file, and the re-pinned chain count
#: pin files.
CARD_EXCEPTION_FILES = tuple(sorted((
    MANIFEST_REL,
    r52.DRIVER_REL_NESTED,
    r52.DRIVER_REL_OUTER,
    MY_TEST_REL,
    *CHAIN_COUNT_PIN_FILES,
)))

# re-pinned 533 -> 536 by Round 5 card 11 (TASK-20260925-r5-run-r51-
# topology-fraction-fit; +3 entries: the dual-tree driver
# helpers/benchmarking/staged_r51_topology_fraction_fit.py, its nested
# copy, and the test file tests/test_staged_r51_topology_fraction_fit.py)
# re-pinned 536 -> 537 by Round 5 card 11 test-author (TASK-20260925-r5-run-r51-topology-fraction-fit-test-author; +1 entry: the card-11 test-author register file tests/test_staged_r51_topology_fraction_fit_contract_pins.py).
# re-pinned 537 -> 540 by Round 5 card 12 (TASK-20260925-r5-run-r52-
# central-excitation-necessity; +3 entries: the dual-tree driver
# dev/mixing2006/helpers/benchmarking/staged_r52_central_excitation_
# necessity.py, its outer mirror helpers/benchmarking/
# staged_r52_central_excitation_necessity.py, and the test file
# tests/test_staged_r52_central_excitation_necessity.py; the card's
# pre-registered 523 -> 526 literal is noted, the operative growth
# followed)
# re-pinned 540 -> 541 by Round 5 card 12 test-author (TASK-20260925-r5-run-r52-central-excitation-necessity-test-author; +1 entry: the card-12 test-author register file tests/test_staged_r52_central_excitation_necessity_contract_pins.py).
# re-pinned 541 -> 544 by Round 5 card 13 (TASK-20260925-r5-run-r53-multiple-neutral-excitation; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r53_multiple_neutral_excitation.py, its outer mirror helpers/benchmarking/staged_r53_multiple_neutral_excitation.py, and the test file tests/test_staged_r53_multiple_neutral_excitation.py; the card's pre-registered 526 -> 529 literal is noted, the operative growth followed).
# re-pinned 544 -> 547 by Round 5 card 14 (TASK-20260925-r5-run-r54-local-ledger-audit; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r54_local_ledger_audit.py, its outer mirror helpers/benchmarking/staged_r54_local_ledger_audit.py, and the test file tests/test_staged_r54_local_ledger_audit.py; the card's pre-registered 529 -> 532 literal is noted, the operative growth followed).
# re-pinned 547 -> 548 by Round 5 card 13 test-author (TASK-20260925-r5-run-r53-multiple-neutral-excitation-test-author; +1 entry: the card-13 test-author register file tests/test_staged_r53_multiple_neutral_excitation_contract_pins.py).
# re-pinned 548 -> 551 by Round 5 card 15 (TASK-20260925-r5-run-r55-frozen-topology-validation; +3 entries: the dual-tree driver dev/mixing2006/helpers/benchmarking/staged_r55_frozen_topology_validation.py, its outer mirror helpers/benchmarking/staged_r55_frozen_topology_validation.py, and the test file tests/test_staged_r55_frozen_topology_validation.py; the card's pre-registered 532 -> 535 literal is noted, the operative growth followed).
INVENTORY_COUNT = _inventory_count()
M8_HASH = (
    "358989d84cb03cffbb0120c64992b1bd95ca732601135e0aa222a6dda69ec0b1")
KERNEL_BOUNDARY = 2.680194110727528


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_of(rel):
    return (r52.MONO_ROOT / rel).read_text(encoding="utf-8")




def test_layout_sync():
    """The layout sync (the subprocess check from the monorepo root;
    ``logic_drift = 0``)."""
    checker = r52.MONO_ROOT / "helpers" / "benchmarking" / \
        "check_helpers_layout_sync.py"
    probe = subprocess.run(
        [sys.executable, str(checker)],
        capture_output=True, text=True, cwd=str(r52.MONO_ROOT))
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "logic_drift=0" in probe.stdout
