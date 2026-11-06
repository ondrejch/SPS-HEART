"""Register of the Round 6 flavor runs R61 to R65 (index
TASK-20260928-round6-index): the five specs against the seed ledger and
the plan's run definitions, dual-tree byte identity of the shared
machinery, the flattened flavor ledger against the per-event weights, a
reduced serial smoke of R61, and the sealed archives when present."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests._sealed_archive import sealed_mismatches

NESTED_ROOT = Path(__file__).resolve().parents[1]
MONO_ROOT = NESTED_ROOT.parents[1] if NESTED_ROOT.parent.name == "dev" else NESTED_ROOT
HELPERS = NESTED_ROOT / "helpers" / "benchmarking"
RUN_FILES = {
    "r61": "staged_r61_pair_strangeness_fit.py", "r62": "staged_r62_associated_strangeness_fit.py",
    "r63": "staged_r63_vector_hidden_strangeness_fit.py", "r64": "staged_r64_baryon_pair_fit.py",
    "r65": "staged_r65_flavor_closure_validation.py",
    "p7": "staged_r7_p7_joint_flavor_refinement.py",
}
SHARED = ("run_contract.py", "run_observables.py", "flavor_fit_core.py", "flavor_run_driver.py")


def _load(name: str):
    if str(HELPERS) not in sys.path:
        sys.path.insert(0, str(HELPERS))
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), HELPERS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def specs():
    return {key: _load(name).SPEC for key, name in RUN_FILES.items()}


def test_dual_tree_byte_identity():
    outer = MONO_ROOT / "helpers" / "benchmarking"
    if outer.resolve() == HELPERS.resolve():
        pytest.skip("monorepo mirror not present")
    for name in SHARED + tuple(RUN_FILES.values()):
        assert (outer / name).read_bytes() == (HELPERS / name).read_bytes(), name


def test_specs_match_the_ledger_and_the_plan(specs):
    ledger = json.loads((NESTED_ROOT / "config" / "v2_singlet" / "seed_ledger.json").read_text())
    blocks = {b["run"]: b for b in ledger["blocks"]}
    for key, spec in specs.items():
        entry = blocks[spec.ledger_run]
        assert (entry["raw_first"], entry["raw_last"], entry["partition_base"]) == (spec.raw_first, spec.raw_last, spec.partition_base), key
        assert spec.raw_last - spec.raw_first + 1 == spec.replicas, key
        assert 20303900 <= spec.smoke_seed <= 20303919 and spec.smoke_partition >= 2_900_000_000
    assert specs["r61"].free == ("gamma_s_had",) and specs["r61"].statistics[0]["species"] == "k_minus"
    assert specs["r62"].free == ("assoc_norm",) and specs["r62"].conditional_trigger["readout_species"] == "k_plus"
    assert specs["r63"].free == ("r_v",) and "r_v" in specs["r63"].priors
    assert specs["r64"].free == ("r_bbar",) and specs["r64"].statistics[0]["species"] == "antiproton"
    assert specs["r65"].validation_only and specs["r65"].free == () and (specs["r65"].replicas, specs["r65"].events_per_replica) == (20, 250_000)
    inherited = {p for spec in specs.values() for p, _ in spec.inherit}
    assert inherited == {"gamma_s_had", "assoc_norm", "r_v", "r_bbar"}
    partitions = [spec.partition_base for spec in specs.values()]
    assert len(set(partitions)) == 6
    assert specs["p7"].free == ("gamma_s_had", "r_bbar") and len(specs["p7"].statistics) == 3


def test_flattened_ledger_matches_per_event_weights():
    FC = _load("flavor_fit_core.py")
    from na49_mixing import topology_runner as TR
    from na49_mixing.flavor_weights import FlavorParams, ledger_log_weight
    from na49_mixing.rng_substreams import derive_substreams
    top = {"fractions": {"sd_projectile": 0.1, "sd_target": 0.1, "dd": 0.05, "nd": 0.75}, "epsilon": 0.0, "b_sd": 8.504140762083612}
    n = 60
    slot = FC.generate_mixture_slot(20303902, n, 2_920_000_000, top, None)
    target = FlavorParams(gamma_s_had=0.6, r_bbar=0.5, r_v=1.3)
    flat = FC.flavor_log_weights(slot["arrays"], n, FlavorParams(), target)
    catalog, fams = FC._load_physics()
    params = TR.TopologyNominalParams(epsilon=0.0, b_sd=8.504140762083612, class_fractions=top["fractions"],
                                      remnant_weights={"intact_proton": 0.5, "low_mass_baryonic_resonance": 0.5, "high_mass_baryonic_continuum": 0.0}, p_n2=0.0)
    for i in range(n):
        subs = derive_substreams(20303902, 0, 2_920_000_000 + i)
        rngs = {f"rng_{l}": np.random.default_rng(subs.stream_for(l)) for l in ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")}
        label = TR.draw_topology(top["fractions"], 0.0, rngs["rng_topology_selection"])
        ledger: list = []
        TR.run_topology(label, params, catalog, fams, **rngs, ledger=ledger)
        assert abs(ledger_log_weight(ledger, None, target) - flat[i]) < 1e-5
    assert np.all(FC.flavor_log_weights(slot["arrays"], n, FlavorParams(), FlavorParams()) == 0.0)


def test_reduced_smoke_of_r61(tmp_path, monkeypatch):
    driver = _load("flavor_run_driver.py")
    spec = _load(RUN_FILES["r61"]).SPEC
    monkeypatch.setattr(driver.RC, "run_replicas", lambda worker, jobs, max_workers=None: [worker(*job) for job in jobs])
    small = spec.__class__(**{**spec.__dict__, "smoke_events": 80})
    driver.step_smoke(small, tmp_path / "r61_smoke")
    report = json.loads((tmp_path / "r61_smoke" / "smoke_fit_report.json").read_text())
    assert set(report["fitted"]) == {"gamma_s_had", "r_v", "r_bbar", "assoc_norm"}
    assert report["statistics"][0]["bins"] == 148
    assert report["readout_at_optimum"]["k_minus"]["bins"] == 148
    assert "pt_moments_held_out" in report["readout_at_optimum"]


@pytest.mark.parametrize("key", list(RUN_FILES))
def test_sealed_archive_on_disk(specs, key):
    spec = specs[key]
    run_dir = MONO_ROOT / "dev" / "00runs" / spec.run_id
    if not (run_dir / "seal_manifest.json").is_file():
        pytest.skip(f"{spec.run_id} is not sealed")
    assert sealed_mismatches(run_dir) == []
    if (run_dir / "not_triggered.json").is_file():
        return
    report = json.loads((run_dir / "report.json").read_text())
    assert report["run"] == spec.run_id and set(report["fitted"]) == {"gamma_s_had", "r_v", "r_bbar", "assoc_norm"}
    gate = json.loads((run_dir / spec.gate_record).read_text())
    assert gate["verdict"] in ("CONSTRAINED", "NOT_CONSTRAINED", "VALIDATION")


def test_mixture_slot_keeps_the_diffractive_split_off_the_fitted_nd_weights():
    """The R51c substrate carries a nonzero continuum weight for the
    non-diffractive remnants; the un-excited single-diffraction line keeps
    the frozen split (the defect that stopped the R61 generation on
    2026-09-29 with generate_sd's continuum guard)."""
    FC = _load("flavor_fit_core.py")
    top = {"fractions": {"sd_projectile": 0.5, "sd_target": 0.5, "dd": 0.0, "nd": 0.0}, "epsilon": -0.49993, "b_sd": 2.3105,
           "remnant_weights": {"intact_proton": 0.00018851, "low_mass_baryonic_resonance": 0.437867, "high_mass_baryonic_continuum": 0.561944}}
    slot = FC.generate_mixture_slot(20303902, 24, 2_920_100_000, top, None)
    census = slot["class_census"]
    assert census["nd"] == 0 and census["sd_projectile"] + census["sd_target"] == 24
    assert slot["topology"]["diffractive_split"] == FC.DIFFRACTIVE_SPLIT
    assert slot["topology"]["remnant_weights_nd"] == top["remnant_weights"]
